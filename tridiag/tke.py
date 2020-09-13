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




__kernel void builtinzhreplicate_f64zireplicate_16615(__global
                                                      unsigned char *mem_16611,
                                                      int32_t num_elems_16612,
                                                      double val_16613)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_16615;
    int32_t replicate_ltid_16616;
    int32_t replicate_gid_16617;
    
    replicate_gtid_16615 = get_global_id(0);
    replicate_ltid_16616 = get_local_id(0);
    replicate_gid_16617 = get_group_id(0);
    if (slt32(replicate_gtid_16615, num_elems_16612)) {
        ((__global double *) mem_16611)[replicate_gtid_16615] = val_16613;
    }
    
  error_0:
    return;
}
__kernel void builtinzhreplicate_i32zireplicate_16606(__global
                                                      unsigned char *mem_16602,
                                                      int32_t num_elems_16603,
                                                      int32_t val_16604)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_16606;
    int32_t replicate_ltid_16607;
    int32_t replicate_gid_16608;
    
    replicate_gtid_16606 = get_global_id(0);
    replicate_ltid_16607 = get_local_id(0);
    replicate_gid_16608 = get_group_id(0);
    if (slt32(replicate_gtid_16606, num_elems_16603)) {
        ((__global int32_t *) mem_16602)[replicate_gtid_16606] = val_16604;
    }
    
  error_0:
    return;
}
__kernel void tridagNestedziscan_stage1_15542(__global int *global_failure,
                                              __local volatile
                                              int64_t *scan_arr_mem_16908_backing_aligned_0,
                                              __local volatile
                                              int64_t *scan_arr_mem_16906_backing_aligned_1,
                                              int32_t n_13843, int32_t m_13844,
                                              int32_t m_13848, __global
                                              unsigned char *c_mem_16411,
                                              __global unsigned char *mem_16488,
                                              __global unsigned char *mem_16505,
                                              __global unsigned char *mem_16515,
                                              __global unsigned char *mem_16520,
                                              int32_t num_threads_16900)
{
    #define segscan_group_sizze_16278 (tridagNestedzisegscan_group_sizze_15536)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16908_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16908_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16906_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16906_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16901;
    int32_t local_tid_16902;
    int32_t group_sizze_16905;
    int32_t wave_sizze_16904;
    int32_t group_tid_16903;
    
    global_tid_16901 = get_global_id(0);
    local_tid_16902 = get_local_id(0);
    group_sizze_16905 = get_local_size(0);
    wave_sizze_16904 = LOCKSTEP_WIDTH;
    group_tid_16903 = get_group_id(0);
    
    int32_t phys_tid_15542;
    
    phys_tid_15542 = global_tid_16901;
    
    __local char *scan_arr_mem_16906;
    __local char *scan_arr_mem_16908;
    
    scan_arr_mem_16906 = (__local char *) scan_arr_mem_16906_backing_0;
    scan_arr_mem_16908 = (__local char *) scan_arr_mem_16908_backing_1;
    
    double x_16283;
    double x_16284;
    double x_16285;
    double x_16286;
    
    x_16283 = 0.0;
    x_16284 = 1.0;
    for (int32_t j_16910 = 0; j_16910 < sdiv_up32(n_13843 * m_13844,
                                                  num_threads_16900);
         j_16910++) {
        int32_t chunk_offset_16911 = segscan_group_sizze_16278 * j_16910 +
                group_tid_16903 * (segscan_group_sizze_16278 *
                                   sdiv_up32(n_13843 * m_13844,
                                             num_threads_16900));
        int32_t flat_idx_16912 = chunk_offset_16911 + local_tid_16902;
        int32_t gtid_15531 = squot32(flat_idx_16912, m_13844);
        int32_t gtid_15541 = flat_idx_16912 - squot32(flat_idx_16912, m_13844) *
                m_13844;
        
        // threads in bounds read input
        {
            if (slt32(gtid_15531, n_13843) && slt32(gtid_15541, m_13844)) {
                int32_t x_16295 = sub32(m_13844, gtid_15541);
                int32_t i_16296 = sub32(x_16295, 1);
                bool cond_16297 = slt32(0, gtid_15541);
                double res_16298;
                double res_16299;
                
                if (cond_16297) {
                    double x_16300 = ((__global
                                       double *) mem_16505)[gtid_15531 *
                                                            m_13844 + i_16296];
                    double y_16301 = ((__global
                                       double *) mem_16488)[gtid_15531 *
                                                            m_13844 + i_16296];
                    double res_16302 = x_16300 / y_16301;
                    double x_16303 = ((__global
                                       double *) c_mem_16411)[gtid_15531 *
                                                              m_13848 +
                                                              i_16296];
                    double y_16304 = x_16303 / y_16301;
                    double res_16305 = 0.0 - y_16304;
                    
                    res_16298 = res_16302;
                    res_16299 = res_16305;
                } else {
                    res_16298 = 0.0;
                    res_16299 = 1.0;
                }
                // write to-scan values to parameters
                {
                    x_16285 = res_16298;
                    x_16286 = res_16299;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_15531, n_13843) && slt32(gtid_15541,
                                                          m_13844))) {
                    x_16285 = 0.0;
                    x_16286 = 1.0;
                }
            }
            // combine with carry and write to local memory
            {
                double y_16287 = x_16283 * x_16286;
                double res_16288 = x_16285 + y_16287;
                double res_16289 = x_16284 * x_16286;
                
                ((__local double *) scan_arr_mem_16906)[local_tid_16902] =
                    res_16288;
                ((__local double *) scan_arr_mem_16908)[local_tid_16902] =
                    res_16289;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            double x_16913;
            double x_16914;
            double x_16915;
            double x_16916;
            double x_16920;
            double x_16921;
            double x_16922;
            double x_16923;
            int32_t skip_threads_16927;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_16902, segscan_group_sizze_16278)) {
                    x_16915 = ((volatile __local
                                double *) scan_arr_mem_16906)[local_tid_16902];
                    x_16916 = ((volatile __local
                                double *) scan_arr_mem_16908)[local_tid_16902];
                    if ((local_tid_16902 - squot32(local_tid_16902, 32) * 32) ==
                        0) {
                        x_16913 = x_16915;
                        x_16914 = x_16916;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_16927 = 1;
                while (slt32(skip_threads_16927, 32)) {
                    if (sle32(skip_threads_16927, local_tid_16902 -
                              squot32(local_tid_16902, 32) * 32) &&
                        slt32(local_tid_16902, segscan_group_sizze_16278)) {
                        // read operands
                        {
                            x_16913 = ((volatile __local
                                        double *) scan_arr_mem_16906)[local_tid_16902 -
                                                                      skip_threads_16927];
                            x_16914 = ((volatile __local
                                        double *) scan_arr_mem_16908)[local_tid_16902 -
                                                                      skip_threads_16927];
                        }
                        // perform operation
                        {
                            bool inactive_16928 = slt32(srem32(local_tid_16902 +
                                                               chunk_offset_16911,
                                                               m_13844),
                                                        local_tid_16902 +
                                                        chunk_offset_16911 -
                                                        (local_tid_16902 -
                                                         skip_threads_16927 +
                                                         chunk_offset_16911));
                            
                            if (inactive_16928) {
                                x_16913 = x_16915;
                                x_16914 = x_16916;
                            }
                            if (!inactive_16928) {
                                double y_16917 = x_16913 * x_16916;
                                double res_16918 = x_16915 + y_16917;
                                double res_16919 = x_16914 * x_16916;
                                
                                x_16913 = res_16918;
                                x_16914 = res_16919;
                            }
                        }
                    }
                    if (sle32(wave_sizze_16904, skip_threads_16927)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_16927, local_tid_16902 -
                              squot32(local_tid_16902, 32) * 32) &&
                        slt32(local_tid_16902, segscan_group_sizze_16278)) {
                        // write result
                        {
                            ((volatile __local
                              double *) scan_arr_mem_16906)[local_tid_16902] =
                                x_16913;
                            x_16915 = x_16913;
                            ((volatile __local
                              double *) scan_arr_mem_16908)[local_tid_16902] =
                                x_16914;
                            x_16916 = x_16914;
                        }
                    }
                    if (sle32(wave_sizze_16904, skip_threads_16927)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_16927 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_16902 - squot32(local_tid_16902, 32) * 32) ==
                    31 && slt32(local_tid_16902, segscan_group_sizze_16278)) {
                    ((volatile __local
                      double *) scan_arr_mem_16906)[squot32(local_tid_16902,
                                                            32)] = x_16913;
                    ((volatile __local
                      double *) scan_arr_mem_16908)[squot32(local_tid_16902,
                                                            32)] = x_16914;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_16929;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_16902, 32) == 0 &&
                        slt32(local_tid_16902, segscan_group_sizze_16278)) {
                        x_16922 = ((volatile __local
                                    double *) scan_arr_mem_16906)[local_tid_16902];
                        x_16923 = ((volatile __local
                                    double *) scan_arr_mem_16908)[local_tid_16902];
                        if ((local_tid_16902 - squot32(local_tid_16902, 32) *
                             32) == 0) {
                            x_16920 = x_16922;
                            x_16921 = x_16923;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_16929 = 1;
                    while (slt32(skip_threads_16929, 32)) {
                        if (sle32(skip_threads_16929, local_tid_16902 -
                                  squot32(local_tid_16902, 32) * 32) &&
                            (squot32(local_tid_16902, 32) == 0 &&
                             slt32(local_tid_16902,
                                   segscan_group_sizze_16278))) {
                            // read operands
                            {
                                x_16920 = ((volatile __local
                                            double *) scan_arr_mem_16906)[local_tid_16902 -
                                                                          skip_threads_16929];
                                x_16921 = ((volatile __local
                                            double *) scan_arr_mem_16908)[local_tid_16902 -
                                                                          skip_threads_16929];
                            }
                            // perform operation
                            {
                                bool inactive_16930 =
                                     slt32(srem32(local_tid_16902 * 32 + 32 -
                                                  1 + chunk_offset_16911,
                                                  m_13844), local_tid_16902 *
                                           32 + 32 - 1 + chunk_offset_16911 -
                                           ((local_tid_16902 -
                                             skip_threads_16929) * 32 + 32 - 1 +
                                            chunk_offset_16911));
                                
                                if (inactive_16930) {
                                    x_16920 = x_16922;
                                    x_16921 = x_16923;
                                }
                                if (!inactive_16930) {
                                    double y_16924 = x_16920 * x_16923;
                                    double res_16925 = x_16922 + y_16924;
                                    double res_16926 = x_16921 * x_16923;
                                    
                                    x_16920 = res_16925;
                                    x_16921 = res_16926;
                                }
                            }
                        }
                        if (sle32(wave_sizze_16904, skip_threads_16929)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_16929, local_tid_16902 -
                                  squot32(local_tid_16902, 32) * 32) &&
                            (squot32(local_tid_16902, 32) == 0 &&
                             slt32(local_tid_16902,
                                   segscan_group_sizze_16278))) {
                            // write result
                            {
                                ((volatile __local
                                  double *) scan_arr_mem_16906)[local_tid_16902] =
                                    x_16920;
                                x_16922 = x_16920;
                                ((volatile __local
                                  double *) scan_arr_mem_16908)[local_tid_16902] =
                                    x_16921;
                                x_16923 = x_16921;
                            }
                        }
                        if (sle32(wave_sizze_16904, skip_threads_16929)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_16929 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_16902, 32) == 0 ||
                      !slt32(local_tid_16902, segscan_group_sizze_16278))) {
                    // read operands
                    {
                        x_16915 = x_16913;
                        x_16916 = x_16914;
                        x_16913 = ((__local
                                    double *) scan_arr_mem_16906)[squot32(local_tid_16902,
                                                                          32) -
                                                                  1];
                        x_16914 = ((__local
                                    double *) scan_arr_mem_16908)[squot32(local_tid_16902,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        bool inactive_16931 = slt32(srem32(local_tid_16902 +
                                                           chunk_offset_16911,
                                                           m_13844),
                                                    local_tid_16902 +
                                                    chunk_offset_16911 -
                                                    (squot32(local_tid_16902,
                                                             32) * 32 - 1 +
                                                     chunk_offset_16911));
                        
                        if (inactive_16931) {
                            x_16913 = x_16915;
                            x_16914 = x_16916;
                        }
                        if (!inactive_16931) {
                            double y_16917 = x_16913 * x_16916;
                            double res_16918 = x_16915 + y_16917;
                            double res_16919 = x_16914 * x_16916;
                            
                            x_16913 = res_16918;
                            x_16914 = res_16919;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          double *) scan_arr_mem_16906)[local_tid_16902] =
                            x_16913;
                        ((__local
                          double *) scan_arr_mem_16908)[local_tid_16902] =
                            x_16914;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_16902, 32) == 0) {
                    ((__local double *) scan_arr_mem_16906)[local_tid_16902] =
                        x_16915;
                    ((__local double *) scan_arr_mem_16908)[local_tid_16902] =
                        x_16916;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_15531, n_13843) && slt32(gtid_15541, m_13844)) {
                    ((__global double *) mem_16515)[gtid_15531 * m_13844 +
                                                    gtid_15541] = ((__local
                                                                    double *) scan_arr_mem_16906)[local_tid_16902];
                    ((__global double *) mem_16520)[gtid_15531 * m_13844 +
                                                    gtid_15541] = ((__local
                                                                    double *) scan_arr_mem_16908)[local_tid_16902];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_16932 = slt32(srem32(chunk_offset_16911 +
                                                          segscan_group_sizze_16278,
                                                          m_13844),
                                                   chunk_offset_16911 +
                                                   segscan_group_sizze_16278 -
                                                   (chunk_offset_16911 +
                                                    segscan_group_sizze_16278 -
                                                    1));
                bool should_load_carry_16933 = local_tid_16902 == 0 &&
                     !crosses_segment_16932;
                
                if (should_load_carry_16933) {
                    x_16283 = ((__local
                                double *) scan_arr_mem_16906)[segscan_group_sizze_16278 -
                                                              1];
                    x_16284 = ((__local
                                double *) scan_arr_mem_16908)[segscan_group_sizze_16278 -
                                                              1];
                }
                if (!should_load_carry_16933) {
                    x_16283 = 0.0;
                    x_16284 = 1.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_16278
}
__kernel void tridagNestedziscan_stage1_15697(__global int *global_failure,
                                              __local volatile
                                              int64_t *scan_arr_mem_16825_backing_aligned_0,
                                              __local volatile
                                              int64_t *scan_arr_mem_16823_backing_aligned_1,
                                              int32_t n_13843, int32_t m_13844,
                                              int32_t m_13850, __global
                                              unsigned char *a_mem_16409,
                                              __global
                                              unsigned char *y_mem_16412,
                                              __global unsigned char *mem_16488,
                                              __global unsigned char *mem_16494,
                                              __global unsigned char *mem_16499,
                                              int32_t num_threads_16817)
{
    #define segscan_group_sizze_16173 (tridagNestedzisegscan_group_sizze_15691)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16825_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16825_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16823_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16823_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16818;
    int32_t local_tid_16819;
    int32_t group_sizze_16822;
    int32_t wave_sizze_16821;
    int32_t group_tid_16820;
    
    global_tid_16818 = get_global_id(0);
    local_tid_16819 = get_local_id(0);
    group_sizze_16822 = get_local_size(0);
    wave_sizze_16821 = LOCKSTEP_WIDTH;
    group_tid_16820 = get_group_id(0);
    
    int32_t phys_tid_15697;
    
    phys_tid_15697 = global_tid_16818;
    
    __local char *scan_arr_mem_16823;
    __local char *scan_arr_mem_16825;
    
    scan_arr_mem_16823 = (__local char *) scan_arr_mem_16823_backing_0;
    scan_arr_mem_16825 = (__local char *) scan_arr_mem_16825_backing_1;
    
    double x_16178;
    double x_16179;
    double x_16180;
    double x_16181;
    
    x_16178 = 0.0;
    x_16179 = 1.0;
    for (int32_t j_16827 = 0; j_16827 < sdiv_up32(n_13843 * m_13844,
                                                  num_threads_16817);
         j_16827++) {
        int32_t chunk_offset_16828 = segscan_group_sizze_16173 * j_16827 +
                group_tid_16820 * (segscan_group_sizze_16173 *
                                   sdiv_up32(n_13843 * m_13844,
                                             num_threads_16817));
        int32_t flat_idx_16829 = chunk_offset_16828 + local_tid_16819;
        int32_t gtid_15686 = squot32(flat_idx_16829, m_13844);
        int32_t gtid_15696 = flat_idx_16829 - squot32(flat_idx_16829, m_13844) *
                m_13844;
        
        // threads in bounds read input
        {
            if (slt32(gtid_15686, n_13843) && slt32(gtid_15696, m_13844)) {
                bool cond_16192 = slt32(0, gtid_15696);
                double res_16193;
                
                if (cond_16192) {
                    double x_elem_16190 = ((__global
                                            double *) y_mem_16412)[gtid_15686 *
                                                                   m_13850 +
                                                                   gtid_15696];
                    
                    res_16193 = x_elem_16190;
                } else {
                    res_16193 = 0.0;
                }
                
                double res_16194;
                
                if (cond_16192) {
                    double x_elem_16191 = ((__global
                                            double *) a_mem_16409)[gtid_15686 *
                                                                   m_13844 +
                                                                   gtid_15696];
                    int32_t i_16195 = sub32(gtid_15696, 1);
                    double y_16196 = ((__global
                                       double *) mem_16488)[gtid_15686 *
                                                            m_13844 + i_16195];
                    double y_16197 = x_elem_16191 / y_16196;
                    double res_16198 = 0.0 - y_16197;
                    
                    res_16194 = res_16198;
                } else {
                    res_16194 = 1.0;
                }
                // write to-scan values to parameters
                {
                    x_16180 = res_16193;
                    x_16181 = res_16194;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_15686, n_13843) && slt32(gtid_15696,
                                                          m_13844))) {
                    x_16180 = 0.0;
                    x_16181 = 1.0;
                }
            }
            // combine with carry and write to local memory
            {
                double y_16182 = x_16178 * x_16181;
                double res_16183 = x_16180 + y_16182;
                double res_16184 = x_16179 * x_16181;
                
                ((__local double *) scan_arr_mem_16823)[local_tid_16819] =
                    res_16183;
                ((__local double *) scan_arr_mem_16825)[local_tid_16819] =
                    res_16184;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            double x_16830;
            double x_16831;
            double x_16832;
            double x_16833;
            double x_16837;
            double x_16838;
            double x_16839;
            double x_16840;
            int32_t skip_threads_16844;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_16819, segscan_group_sizze_16173)) {
                    x_16832 = ((volatile __local
                                double *) scan_arr_mem_16823)[local_tid_16819];
                    x_16833 = ((volatile __local
                                double *) scan_arr_mem_16825)[local_tid_16819];
                    if ((local_tid_16819 - squot32(local_tid_16819, 32) * 32) ==
                        0) {
                        x_16830 = x_16832;
                        x_16831 = x_16833;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_16844 = 1;
                while (slt32(skip_threads_16844, 32)) {
                    if (sle32(skip_threads_16844, local_tid_16819 -
                              squot32(local_tid_16819, 32) * 32) &&
                        slt32(local_tid_16819, segscan_group_sizze_16173)) {
                        // read operands
                        {
                            x_16830 = ((volatile __local
                                        double *) scan_arr_mem_16823)[local_tid_16819 -
                                                                      skip_threads_16844];
                            x_16831 = ((volatile __local
                                        double *) scan_arr_mem_16825)[local_tid_16819 -
                                                                      skip_threads_16844];
                        }
                        // perform operation
                        {
                            bool inactive_16845 = slt32(srem32(local_tid_16819 +
                                                               chunk_offset_16828,
                                                               m_13844),
                                                        local_tid_16819 +
                                                        chunk_offset_16828 -
                                                        (local_tid_16819 -
                                                         skip_threads_16844 +
                                                         chunk_offset_16828));
                            
                            if (inactive_16845) {
                                x_16830 = x_16832;
                                x_16831 = x_16833;
                            }
                            if (!inactive_16845) {
                                double y_16834 = x_16830 * x_16833;
                                double res_16835 = x_16832 + y_16834;
                                double res_16836 = x_16831 * x_16833;
                                
                                x_16830 = res_16835;
                                x_16831 = res_16836;
                            }
                        }
                    }
                    if (sle32(wave_sizze_16821, skip_threads_16844)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_16844, local_tid_16819 -
                              squot32(local_tid_16819, 32) * 32) &&
                        slt32(local_tid_16819, segscan_group_sizze_16173)) {
                        // write result
                        {
                            ((volatile __local
                              double *) scan_arr_mem_16823)[local_tid_16819] =
                                x_16830;
                            x_16832 = x_16830;
                            ((volatile __local
                              double *) scan_arr_mem_16825)[local_tid_16819] =
                                x_16831;
                            x_16833 = x_16831;
                        }
                    }
                    if (sle32(wave_sizze_16821, skip_threads_16844)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_16844 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_16819 - squot32(local_tid_16819, 32) * 32) ==
                    31 && slt32(local_tid_16819, segscan_group_sizze_16173)) {
                    ((volatile __local
                      double *) scan_arr_mem_16823)[squot32(local_tid_16819,
                                                            32)] = x_16830;
                    ((volatile __local
                      double *) scan_arr_mem_16825)[squot32(local_tid_16819,
                                                            32)] = x_16831;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_16846;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_16819, 32) == 0 &&
                        slt32(local_tid_16819, segscan_group_sizze_16173)) {
                        x_16839 = ((volatile __local
                                    double *) scan_arr_mem_16823)[local_tid_16819];
                        x_16840 = ((volatile __local
                                    double *) scan_arr_mem_16825)[local_tid_16819];
                        if ((local_tid_16819 - squot32(local_tid_16819, 32) *
                             32) == 0) {
                            x_16837 = x_16839;
                            x_16838 = x_16840;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_16846 = 1;
                    while (slt32(skip_threads_16846, 32)) {
                        if (sle32(skip_threads_16846, local_tid_16819 -
                                  squot32(local_tid_16819, 32) * 32) &&
                            (squot32(local_tid_16819, 32) == 0 &&
                             slt32(local_tid_16819,
                                   segscan_group_sizze_16173))) {
                            // read operands
                            {
                                x_16837 = ((volatile __local
                                            double *) scan_arr_mem_16823)[local_tid_16819 -
                                                                          skip_threads_16846];
                                x_16838 = ((volatile __local
                                            double *) scan_arr_mem_16825)[local_tid_16819 -
                                                                          skip_threads_16846];
                            }
                            // perform operation
                            {
                                bool inactive_16847 =
                                     slt32(srem32(local_tid_16819 * 32 + 32 -
                                                  1 + chunk_offset_16828,
                                                  m_13844), local_tid_16819 *
                                           32 + 32 - 1 + chunk_offset_16828 -
                                           ((local_tid_16819 -
                                             skip_threads_16846) * 32 + 32 - 1 +
                                            chunk_offset_16828));
                                
                                if (inactive_16847) {
                                    x_16837 = x_16839;
                                    x_16838 = x_16840;
                                }
                                if (!inactive_16847) {
                                    double y_16841 = x_16837 * x_16840;
                                    double res_16842 = x_16839 + y_16841;
                                    double res_16843 = x_16838 * x_16840;
                                    
                                    x_16837 = res_16842;
                                    x_16838 = res_16843;
                                }
                            }
                        }
                        if (sle32(wave_sizze_16821, skip_threads_16846)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_16846, local_tid_16819 -
                                  squot32(local_tid_16819, 32) * 32) &&
                            (squot32(local_tid_16819, 32) == 0 &&
                             slt32(local_tid_16819,
                                   segscan_group_sizze_16173))) {
                            // write result
                            {
                                ((volatile __local
                                  double *) scan_arr_mem_16823)[local_tid_16819] =
                                    x_16837;
                                x_16839 = x_16837;
                                ((volatile __local
                                  double *) scan_arr_mem_16825)[local_tid_16819] =
                                    x_16838;
                                x_16840 = x_16838;
                            }
                        }
                        if (sle32(wave_sizze_16821, skip_threads_16846)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_16846 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_16819, 32) == 0 ||
                      !slt32(local_tid_16819, segscan_group_sizze_16173))) {
                    // read operands
                    {
                        x_16832 = x_16830;
                        x_16833 = x_16831;
                        x_16830 = ((__local
                                    double *) scan_arr_mem_16823)[squot32(local_tid_16819,
                                                                          32) -
                                                                  1];
                        x_16831 = ((__local
                                    double *) scan_arr_mem_16825)[squot32(local_tid_16819,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        bool inactive_16848 = slt32(srem32(local_tid_16819 +
                                                           chunk_offset_16828,
                                                           m_13844),
                                                    local_tid_16819 +
                                                    chunk_offset_16828 -
                                                    (squot32(local_tid_16819,
                                                             32) * 32 - 1 +
                                                     chunk_offset_16828));
                        
                        if (inactive_16848) {
                            x_16830 = x_16832;
                            x_16831 = x_16833;
                        }
                        if (!inactive_16848) {
                            double y_16834 = x_16830 * x_16833;
                            double res_16835 = x_16832 + y_16834;
                            double res_16836 = x_16831 * x_16833;
                            
                            x_16830 = res_16835;
                            x_16831 = res_16836;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          double *) scan_arr_mem_16823)[local_tid_16819] =
                            x_16830;
                        ((__local
                          double *) scan_arr_mem_16825)[local_tid_16819] =
                            x_16831;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_16819, 32) == 0) {
                    ((__local double *) scan_arr_mem_16823)[local_tid_16819] =
                        x_16832;
                    ((__local double *) scan_arr_mem_16825)[local_tid_16819] =
                        x_16833;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_15686, n_13843) && slt32(gtid_15696, m_13844)) {
                    ((__global double *) mem_16494)[gtid_15686 * m_13844 +
                                                    gtid_15696] = ((__local
                                                                    double *) scan_arr_mem_16823)[local_tid_16819];
                    ((__global double *) mem_16499)[gtid_15686 * m_13844 +
                                                    gtid_15696] = ((__local
                                                                    double *) scan_arr_mem_16825)[local_tid_16819];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_16849 = slt32(srem32(chunk_offset_16828 +
                                                          segscan_group_sizze_16173,
                                                          m_13844),
                                                   chunk_offset_16828 +
                                                   segscan_group_sizze_16173 -
                                                   (chunk_offset_16828 +
                                                    segscan_group_sizze_16173 -
                                                    1));
                bool should_load_carry_16850 = local_tid_16819 == 0 &&
                     !crosses_segment_16849;
                
                if (should_load_carry_16850) {
                    x_16178 = ((__local
                                double *) scan_arr_mem_16823)[segscan_group_sizze_16173 -
                                                              1];
                    x_16179 = ((__local
                                double *) scan_arr_mem_16825)[segscan_group_sizze_16173 -
                                                              1];
                }
                if (!should_load_carry_16850) {
                    x_16178 = 0.0;
                    x_16179 = 1.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_16173
}
__kernel void tridagNestedziscan_stage1_15930(__global int *global_failure,
                                              __local volatile
                                              int64_t *scan_arr_mem_16692_backing_aligned_0,
                                              __local volatile
                                              int64_t *scan_arr_mem_16690_backing_aligned_1,
                                              __local volatile
                                              int64_t *scan_arr_mem_16688_backing_aligned_2,
                                              __local volatile
                                              int64_t *scan_arr_mem_16686_backing_aligned_3,
                                              int32_t n_13843, int32_t m_13844,
                                              int32_t m_13846, int32_t m_13848,
                                              __global
                                              unsigned char *a_mem_16409,
                                              __global
                                              unsigned char *b_mem_16410,
                                              __global
                                              unsigned char *c_mem_16411,
                                              __global unsigned char *mem_16467,
                                              __global unsigned char *mem_16472,
                                              __global unsigned char *mem_16477,
                                              __global unsigned char *mem_16482,
                                              int32_t num_threads_16680)
{
    #define segscan_group_sizze_16005 (tridagNestedzisegscan_group_sizze_15924)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16692_backing_3 =
                          (__local volatile
                           char *) scan_arr_mem_16692_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16690_backing_2 =
                          (__local volatile
                           char *) scan_arr_mem_16690_backing_aligned_1;
    __local volatile char *restrict scan_arr_mem_16688_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16688_backing_aligned_2;
    __local volatile char *restrict scan_arr_mem_16686_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16686_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16681;
    int32_t local_tid_16682;
    int32_t group_sizze_16685;
    int32_t wave_sizze_16684;
    int32_t group_tid_16683;
    
    global_tid_16681 = get_global_id(0);
    local_tid_16682 = get_local_id(0);
    group_sizze_16685 = get_local_size(0);
    wave_sizze_16684 = LOCKSTEP_WIDTH;
    group_tid_16683 = get_group_id(0);
    
    int32_t phys_tid_15930;
    
    phys_tid_15930 = global_tid_16681;
    
    __local char *scan_arr_mem_16686;
    __local char *scan_arr_mem_16688;
    __local char *scan_arr_mem_16690;
    __local char *scan_arr_mem_16692;
    
    scan_arr_mem_16686 = (__local char *) scan_arr_mem_16686_backing_0;
    scan_arr_mem_16688 = (__local char *) scan_arr_mem_16688_backing_1;
    scan_arr_mem_16690 = (__local char *) scan_arr_mem_16690_backing_2;
    scan_arr_mem_16692 = (__local char *) scan_arr_mem_16692_backing_3;
    
    double x_16012;
    double x_16013;
    double x_16014;
    double x_16015;
    double x_16016;
    double x_16017;
    double x_16018;
    double x_16019;
    
    x_16012 = 1.0;
    x_16013 = 0.0;
    x_16014 = 0.0;
    x_16015 = 1.0;
    for (int32_t j_16694 = 0; j_16694 < sdiv_up32(n_13843 * m_13844,
                                                  num_threads_16680);
         j_16694++) {
        int32_t chunk_offset_16695 = segscan_group_sizze_16005 * j_16694 +
                group_tid_16683 * (segscan_group_sizze_16005 *
                                   sdiv_up32(n_13843 * m_13844,
                                             num_threads_16680));
        int32_t flat_idx_16696 = chunk_offset_16695 + local_tid_16682;
        int32_t gtid_15919 = squot32(flat_idx_16696, m_13844);
        int32_t gtid_15929 = flat_idx_16696 - squot32(flat_idx_16696, m_13844) *
                m_13844;
        
        // threads in bounds read input
        {
            if (slt32(gtid_15919, n_13843) && slt32(gtid_15929, m_13844)) {
                bool cond_16044 = slt32(0, gtid_15929);
                double res_16045;
                
                if (cond_16044) {
                    res_16045 = 1.0;
                } else {
                    res_16045 = 0.0;
                }
                
                double res_16046;
                
                if (cond_16044) {
                    res_16046 = 0.0;
                } else {
                    res_16046 = 1.0;
                }
                
                double res_16047;
                
                if (cond_16044) {
                    double x_elem_16042 = ((__global
                                            double *) b_mem_16410)[gtid_15919 *
                                                                   m_13846 +
                                                                   gtid_15929];
                    
                    res_16047 = x_elem_16042;
                } else {
                    res_16047 = 1.0;
                }
                
                double res_16048;
                
                if (cond_16044) {
                    double x_elem_16043 = ((__global
                                            double *) a_mem_16409)[gtid_15919 *
                                                                   m_13844 +
                                                                   gtid_15929];
                    int32_t i_16049 = sub32(gtid_15929, 1);
                    double y_16050 = ((__global
                                       double *) c_mem_16411)[gtid_15919 *
                                                              m_13848 +
                                                              i_16049];
                    double y_16051 = x_elem_16043 * y_16050;
                    double res_16052 = 0.0 - y_16051;
                    
                    res_16048 = res_16052;
                } else {
                    res_16048 = 0.0;
                }
                // write to-scan values to parameters
                {
                    x_16016 = res_16047;
                    x_16017 = res_16048;
                    x_16018 = res_16045;
                    x_16019 = res_16046;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_15919, n_13843) && slt32(gtid_15929,
                                                          m_13844))) {
                    x_16016 = 1.0;
                    x_16017 = 0.0;
                    x_16018 = 0.0;
                    x_16019 = 1.0;
                }
            }
            // combine with carry and write to local memory
            {
                double y_16020 = x_16012 * x_16016;
                double value_16021 = 1.0 / y_16020;
                double y_16022 = x_16014 * x_16017;
                double x_16023 = y_16020 + y_16022;
                double res_16024 = value_16021 * x_16023;
                double x_16025 = x_16013 * x_16016;
                double y_16026 = x_16015 * x_16017;
                double x_16027 = x_16025 + y_16026;
                double res_16028 = value_16021 * x_16027;
                double x_16029 = x_16012 * x_16018;
                double y_16030 = x_16014 * x_16019;
                double x_16031 = x_16029 + y_16030;
                double res_16032 = value_16021 * x_16031;
                double x_16033 = x_16013 * x_16018;
                double y_16034 = x_16015 * x_16019;
                double x_16035 = x_16033 + y_16034;
                double res_16036 = value_16021 * x_16035;
                
                ((__local double *) scan_arr_mem_16686)[local_tid_16682] =
                    res_16024;
                ((__local double *) scan_arr_mem_16688)[local_tid_16682] =
                    res_16028;
                ((__local double *) scan_arr_mem_16690)[local_tid_16682] =
                    res_16032;
                ((__local double *) scan_arr_mem_16692)[local_tid_16682] =
                    res_16036;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            double x_16697;
            double x_16698;
            double x_16699;
            double x_16700;
            double x_16701;
            double x_16702;
            double x_16703;
            double x_16704;
            double x_16722;
            double x_16723;
            double x_16724;
            double x_16725;
            double x_16726;
            double x_16727;
            double x_16728;
            double x_16729;
            int32_t skip_threads_16747;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_16682, segscan_group_sizze_16005)) {
                    x_16701 = ((volatile __local
                                double *) scan_arr_mem_16686)[local_tid_16682];
                    x_16702 = ((volatile __local
                                double *) scan_arr_mem_16688)[local_tid_16682];
                    x_16703 = ((volatile __local
                                double *) scan_arr_mem_16690)[local_tid_16682];
                    x_16704 = ((volatile __local
                                double *) scan_arr_mem_16692)[local_tid_16682];
                    if ((local_tid_16682 - squot32(local_tid_16682, 32) * 32) ==
                        0) {
                        x_16697 = x_16701;
                        x_16698 = x_16702;
                        x_16699 = x_16703;
                        x_16700 = x_16704;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_16747 = 1;
                while (slt32(skip_threads_16747, 32)) {
                    if (sle32(skip_threads_16747, local_tid_16682 -
                              squot32(local_tid_16682, 32) * 32) &&
                        slt32(local_tid_16682, segscan_group_sizze_16005)) {
                        // read operands
                        {
                            x_16697 = ((volatile __local
                                        double *) scan_arr_mem_16686)[local_tid_16682 -
                                                                      skip_threads_16747];
                            x_16698 = ((volatile __local
                                        double *) scan_arr_mem_16688)[local_tid_16682 -
                                                                      skip_threads_16747];
                            x_16699 = ((volatile __local
                                        double *) scan_arr_mem_16690)[local_tid_16682 -
                                                                      skip_threads_16747];
                            x_16700 = ((volatile __local
                                        double *) scan_arr_mem_16692)[local_tid_16682 -
                                                                      skip_threads_16747];
                        }
                        // perform operation
                        {
                            bool inactive_16748 = slt32(srem32(local_tid_16682 +
                                                               chunk_offset_16695,
                                                               m_13844),
                                                        local_tid_16682 +
                                                        chunk_offset_16695 -
                                                        (local_tid_16682 -
                                                         skip_threads_16747 +
                                                         chunk_offset_16695));
                            
                            if (inactive_16748) {
                                x_16697 = x_16701;
                                x_16698 = x_16702;
                                x_16699 = x_16703;
                                x_16700 = x_16704;
                            }
                            if (!inactive_16748) {
                                double y_16705 = x_16697 * x_16701;
                                double value_16706 = 1.0 / y_16705;
                                double y_16707 = x_16699 * x_16702;
                                double x_16708 = y_16705 + y_16707;
                                double res_16709 = value_16706 * x_16708;
                                double x_16710 = x_16698 * x_16701;
                                double y_16711 = x_16700 * x_16702;
                                double x_16712 = x_16710 + y_16711;
                                double res_16713 = value_16706 * x_16712;
                                double x_16714 = x_16697 * x_16703;
                                double y_16715 = x_16699 * x_16704;
                                double x_16716 = x_16714 + y_16715;
                                double res_16717 = value_16706 * x_16716;
                                double x_16718 = x_16698 * x_16703;
                                double y_16719 = x_16700 * x_16704;
                                double x_16720 = x_16718 + y_16719;
                                double res_16721 = value_16706 * x_16720;
                                
                                x_16697 = res_16709;
                                x_16698 = res_16713;
                                x_16699 = res_16717;
                                x_16700 = res_16721;
                            }
                        }
                    }
                    if (sle32(wave_sizze_16684, skip_threads_16747)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_16747, local_tid_16682 -
                              squot32(local_tid_16682, 32) * 32) &&
                        slt32(local_tid_16682, segscan_group_sizze_16005)) {
                        // write result
                        {
                            ((volatile __local
                              double *) scan_arr_mem_16686)[local_tid_16682] =
                                x_16697;
                            x_16701 = x_16697;
                            ((volatile __local
                              double *) scan_arr_mem_16688)[local_tid_16682] =
                                x_16698;
                            x_16702 = x_16698;
                            ((volatile __local
                              double *) scan_arr_mem_16690)[local_tid_16682] =
                                x_16699;
                            x_16703 = x_16699;
                            ((volatile __local
                              double *) scan_arr_mem_16692)[local_tid_16682] =
                                x_16700;
                            x_16704 = x_16700;
                        }
                    }
                    if (sle32(wave_sizze_16684, skip_threads_16747)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_16747 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_16682 - squot32(local_tid_16682, 32) * 32) ==
                    31 && slt32(local_tid_16682, segscan_group_sizze_16005)) {
                    ((volatile __local
                      double *) scan_arr_mem_16686)[squot32(local_tid_16682,
                                                            32)] = x_16697;
                    ((volatile __local
                      double *) scan_arr_mem_16688)[squot32(local_tid_16682,
                                                            32)] = x_16698;
                    ((volatile __local
                      double *) scan_arr_mem_16690)[squot32(local_tid_16682,
                                                            32)] = x_16699;
                    ((volatile __local
                      double *) scan_arr_mem_16692)[squot32(local_tid_16682,
                                                            32)] = x_16700;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_16749;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_16682, 32) == 0 &&
                        slt32(local_tid_16682, segscan_group_sizze_16005)) {
                        x_16726 = ((volatile __local
                                    double *) scan_arr_mem_16686)[local_tid_16682];
                        x_16727 = ((volatile __local
                                    double *) scan_arr_mem_16688)[local_tid_16682];
                        x_16728 = ((volatile __local
                                    double *) scan_arr_mem_16690)[local_tid_16682];
                        x_16729 = ((volatile __local
                                    double *) scan_arr_mem_16692)[local_tid_16682];
                        if ((local_tid_16682 - squot32(local_tid_16682, 32) *
                             32) == 0) {
                            x_16722 = x_16726;
                            x_16723 = x_16727;
                            x_16724 = x_16728;
                            x_16725 = x_16729;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_16749 = 1;
                    while (slt32(skip_threads_16749, 32)) {
                        if (sle32(skip_threads_16749, local_tid_16682 -
                                  squot32(local_tid_16682, 32) * 32) &&
                            (squot32(local_tid_16682, 32) == 0 &&
                             slt32(local_tid_16682,
                                   segscan_group_sizze_16005))) {
                            // read operands
                            {
                                x_16722 = ((volatile __local
                                            double *) scan_arr_mem_16686)[local_tid_16682 -
                                                                          skip_threads_16749];
                                x_16723 = ((volatile __local
                                            double *) scan_arr_mem_16688)[local_tid_16682 -
                                                                          skip_threads_16749];
                                x_16724 = ((volatile __local
                                            double *) scan_arr_mem_16690)[local_tid_16682 -
                                                                          skip_threads_16749];
                                x_16725 = ((volatile __local
                                            double *) scan_arr_mem_16692)[local_tid_16682 -
                                                                          skip_threads_16749];
                            }
                            // perform operation
                            {
                                bool inactive_16750 =
                                     slt32(srem32(local_tid_16682 * 32 + 32 -
                                                  1 + chunk_offset_16695,
                                                  m_13844), local_tid_16682 *
                                           32 + 32 - 1 + chunk_offset_16695 -
                                           ((local_tid_16682 -
                                             skip_threads_16749) * 32 + 32 - 1 +
                                            chunk_offset_16695));
                                
                                if (inactive_16750) {
                                    x_16722 = x_16726;
                                    x_16723 = x_16727;
                                    x_16724 = x_16728;
                                    x_16725 = x_16729;
                                }
                                if (!inactive_16750) {
                                    double y_16730 = x_16722 * x_16726;
                                    double value_16731 = 1.0 / y_16730;
                                    double y_16732 = x_16724 * x_16727;
                                    double x_16733 = y_16730 + y_16732;
                                    double res_16734 = value_16731 * x_16733;
                                    double x_16735 = x_16723 * x_16726;
                                    double y_16736 = x_16725 * x_16727;
                                    double x_16737 = x_16735 + y_16736;
                                    double res_16738 = value_16731 * x_16737;
                                    double x_16739 = x_16722 * x_16728;
                                    double y_16740 = x_16724 * x_16729;
                                    double x_16741 = x_16739 + y_16740;
                                    double res_16742 = value_16731 * x_16741;
                                    double x_16743 = x_16723 * x_16728;
                                    double y_16744 = x_16725 * x_16729;
                                    double x_16745 = x_16743 + y_16744;
                                    double res_16746 = value_16731 * x_16745;
                                    
                                    x_16722 = res_16734;
                                    x_16723 = res_16738;
                                    x_16724 = res_16742;
                                    x_16725 = res_16746;
                                }
                            }
                        }
                        if (sle32(wave_sizze_16684, skip_threads_16749)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_16749, local_tid_16682 -
                                  squot32(local_tid_16682, 32) * 32) &&
                            (squot32(local_tid_16682, 32) == 0 &&
                             slt32(local_tid_16682,
                                   segscan_group_sizze_16005))) {
                            // write result
                            {
                                ((volatile __local
                                  double *) scan_arr_mem_16686)[local_tid_16682] =
                                    x_16722;
                                x_16726 = x_16722;
                                ((volatile __local
                                  double *) scan_arr_mem_16688)[local_tid_16682] =
                                    x_16723;
                                x_16727 = x_16723;
                                ((volatile __local
                                  double *) scan_arr_mem_16690)[local_tid_16682] =
                                    x_16724;
                                x_16728 = x_16724;
                                ((volatile __local
                                  double *) scan_arr_mem_16692)[local_tid_16682] =
                                    x_16725;
                                x_16729 = x_16725;
                            }
                        }
                        if (sle32(wave_sizze_16684, skip_threads_16749)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_16749 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_16682, 32) == 0 ||
                      !slt32(local_tid_16682, segscan_group_sizze_16005))) {
                    // read operands
                    {
                        x_16701 = x_16697;
                        x_16702 = x_16698;
                        x_16703 = x_16699;
                        x_16704 = x_16700;
                        x_16697 = ((__local
                                    double *) scan_arr_mem_16686)[squot32(local_tid_16682,
                                                                          32) -
                                                                  1];
                        x_16698 = ((__local
                                    double *) scan_arr_mem_16688)[squot32(local_tid_16682,
                                                                          32) -
                                                                  1];
                        x_16699 = ((__local
                                    double *) scan_arr_mem_16690)[squot32(local_tid_16682,
                                                                          32) -
                                                                  1];
                        x_16700 = ((__local
                                    double *) scan_arr_mem_16692)[squot32(local_tid_16682,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        bool inactive_16751 = slt32(srem32(local_tid_16682 +
                                                           chunk_offset_16695,
                                                           m_13844),
                                                    local_tid_16682 +
                                                    chunk_offset_16695 -
                                                    (squot32(local_tid_16682,
                                                             32) * 32 - 1 +
                                                     chunk_offset_16695));
                        
                        if (inactive_16751) {
                            x_16697 = x_16701;
                            x_16698 = x_16702;
                            x_16699 = x_16703;
                            x_16700 = x_16704;
                        }
                        if (!inactive_16751) {
                            double y_16705 = x_16697 * x_16701;
                            double value_16706 = 1.0 / y_16705;
                            double y_16707 = x_16699 * x_16702;
                            double x_16708 = y_16705 + y_16707;
                            double res_16709 = value_16706 * x_16708;
                            double x_16710 = x_16698 * x_16701;
                            double y_16711 = x_16700 * x_16702;
                            double x_16712 = x_16710 + y_16711;
                            double res_16713 = value_16706 * x_16712;
                            double x_16714 = x_16697 * x_16703;
                            double y_16715 = x_16699 * x_16704;
                            double x_16716 = x_16714 + y_16715;
                            double res_16717 = value_16706 * x_16716;
                            double x_16718 = x_16698 * x_16703;
                            double y_16719 = x_16700 * x_16704;
                            double x_16720 = x_16718 + y_16719;
                            double res_16721 = value_16706 * x_16720;
                            
                            x_16697 = res_16709;
                            x_16698 = res_16713;
                            x_16699 = res_16717;
                            x_16700 = res_16721;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          double *) scan_arr_mem_16686)[local_tid_16682] =
                            x_16697;
                        ((__local
                          double *) scan_arr_mem_16688)[local_tid_16682] =
                            x_16698;
                        ((__local
                          double *) scan_arr_mem_16690)[local_tid_16682] =
                            x_16699;
                        ((__local
                          double *) scan_arr_mem_16692)[local_tid_16682] =
                            x_16700;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_16682, 32) == 0) {
                    ((__local double *) scan_arr_mem_16686)[local_tid_16682] =
                        x_16701;
                    ((__local double *) scan_arr_mem_16688)[local_tid_16682] =
                        x_16702;
                    ((__local double *) scan_arr_mem_16690)[local_tid_16682] =
                        x_16703;
                    ((__local double *) scan_arr_mem_16692)[local_tid_16682] =
                        x_16704;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_15919, n_13843) && slt32(gtid_15929, m_13844)) {
                    ((__global double *) mem_16467)[gtid_15919 * m_13844 +
                                                    gtid_15929] = ((__local
                                                                    double *) scan_arr_mem_16686)[local_tid_16682];
                    ((__global double *) mem_16472)[gtid_15919 * m_13844 +
                                                    gtid_15929] = ((__local
                                                                    double *) scan_arr_mem_16688)[local_tid_16682];
                    ((__global double *) mem_16477)[gtid_15919 * m_13844 +
                                                    gtid_15929] = ((__local
                                                                    double *) scan_arr_mem_16690)[local_tid_16682];
                    ((__global double *) mem_16482)[gtid_15919 * m_13844 +
                                                    gtid_15929] = ((__local
                                                                    double *) scan_arr_mem_16692)[local_tid_16682];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_16752 = slt32(srem32(chunk_offset_16695 +
                                                          segscan_group_sizze_16005,
                                                          m_13844),
                                                   chunk_offset_16695 +
                                                   segscan_group_sizze_16005 -
                                                   (chunk_offset_16695 +
                                                    segscan_group_sizze_16005 -
                                                    1));
                bool should_load_carry_16753 = local_tid_16682 == 0 &&
                     !crosses_segment_16752;
                
                if (should_load_carry_16753) {
                    x_16012 = ((__local
                                double *) scan_arr_mem_16686)[segscan_group_sizze_16005 -
                                                              1];
                    x_16013 = ((__local
                                double *) scan_arr_mem_16688)[segscan_group_sizze_16005 -
                                                              1];
                    x_16014 = ((__local
                                double *) scan_arr_mem_16690)[segscan_group_sizze_16005 -
                                                              1];
                    x_16015 = ((__local
                                double *) scan_arr_mem_16692)[segscan_group_sizze_16005 -
                                                              1];
                }
                if (!should_load_carry_16753) {
                    x_16012 = 1.0;
                    x_16013 = 0.0;
                    x_16014 = 0.0;
                    x_16015 = 1.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_16005
}
__kernel void tridagNestedziscan_stage2_15542(__global int *global_failure,
                                              __local volatile
                                              int64_t *scan_arr_mem_16941_backing_aligned_0,
                                              __local volatile
                                              int64_t *scan_arr_mem_16939_backing_aligned_1,
                                              int32_t n_13843, int32_t m_13844,
                                              __global unsigned char *mem_16515,
                                              __global unsigned char *mem_16520,
                                              int32_t stage1_num_groups_16899,
                                              int32_t num_threads_16900)
{
    #define segscan_group_sizze_16278 (tridagNestedzisegscan_group_sizze_15536)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16941_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16941_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16939_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16939_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16934;
    int32_t local_tid_16935;
    int32_t group_sizze_16938;
    int32_t wave_sizze_16937;
    int32_t group_tid_16936;
    
    global_tid_16934 = get_global_id(0);
    local_tid_16935 = get_local_id(0);
    group_sizze_16938 = get_local_size(0);
    wave_sizze_16937 = LOCKSTEP_WIDTH;
    group_tid_16936 = get_group_id(0);
    
    int32_t phys_tid_15542;
    
    phys_tid_15542 = global_tid_16934;
    
    __local char *scan_arr_mem_16939;
    __local char *scan_arr_mem_16941;
    
    scan_arr_mem_16939 = (__local char *) scan_arr_mem_16939_backing_0;
    scan_arr_mem_16941 = (__local char *) scan_arr_mem_16941_backing_1;
    
    int32_t flat_idx_16943;
    
    flat_idx_16943 = (local_tid_16935 + 1) * (segscan_group_sizze_16278 *
                                              sdiv_up32(n_13843 * m_13844,
                                                        num_threads_16900)) - 1;
    
    int32_t gtid_15531;
    
    gtid_15531 = squot32(flat_idx_16943, m_13844);
    
    int32_t gtid_15541;
    
    gtid_15541 = flat_idx_16943 - squot32(flat_idx_16943, m_13844) * m_13844;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_15531, n_13843) && slt32(gtid_15541, m_13844)) {
            ((__local double *) scan_arr_mem_16939)[local_tid_16935] =
                ((__global double *) mem_16515)[gtid_15531 * m_13844 +
                                                gtid_15541];
            ((__local double *) scan_arr_mem_16941)[local_tid_16935] =
                ((__global double *) mem_16520)[gtid_15531 * m_13844 +
                                                gtid_15541];
        } else {
            ((__local double *) scan_arr_mem_16939)[local_tid_16935] = 0.0;
            ((__local double *) scan_arr_mem_16941)[local_tid_16935] = 1.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    double x_16283;
    double x_16284;
    double x_16285;
    double x_16286;
    double x_16944;
    double x_16945;
    double x_16946;
    double x_16947;
    int32_t skip_threads_16951;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16935, stage1_num_groups_16899)) {
            x_16285 = ((volatile __local
                        double *) scan_arr_mem_16939)[local_tid_16935];
            x_16286 = ((volatile __local
                        double *) scan_arr_mem_16941)[local_tid_16935];
            if ((local_tid_16935 - squot32(local_tid_16935, 32) * 32) == 0) {
                x_16283 = x_16285;
                x_16284 = x_16286;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16951 = 1;
        while (slt32(skip_threads_16951, 32)) {
            if (sle32(skip_threads_16951, local_tid_16935 -
                      squot32(local_tid_16935, 32) * 32) &&
                slt32(local_tid_16935, stage1_num_groups_16899)) {
                // read operands
                {
                    x_16283 = ((volatile __local
                                double *) scan_arr_mem_16939)[local_tid_16935 -
                                                              skip_threads_16951];
                    x_16284 = ((volatile __local
                                double *) scan_arr_mem_16941)[local_tid_16935 -
                                                              skip_threads_16951];
                }
                // perform operation
                {
                    bool inactive_16952 = slt32(srem32((local_tid_16935 + 1) *
                                                       (segscan_group_sizze_16278 *
                                                        sdiv_up32(n_13843 *
                                                                  m_13844,
                                                                  num_threads_16900)) -
                                                       1, m_13844),
                                                (local_tid_16935 + 1) *
                                                (segscan_group_sizze_16278 *
                                                 sdiv_up32(n_13843 * m_13844,
                                                           num_threads_16900)) -
                                                1 - ((local_tid_16935 -
                                                      skip_threads_16951 + 1) *
                                                     (segscan_group_sizze_16278 *
                                                      sdiv_up32(n_13843 *
                                                                m_13844,
                                                                num_threads_16900)) -
                                                     1));
                    
                    if (inactive_16952) {
                        x_16283 = x_16285;
                        x_16284 = x_16286;
                    }
                    if (!inactive_16952) {
                        double y_16287 = x_16283 * x_16286;
                        double res_16288 = x_16285 + y_16287;
                        double res_16289 = x_16284 * x_16286;
                        
                        x_16283 = res_16288;
                        x_16284 = res_16289;
                    }
                }
            }
            if (sle32(wave_sizze_16937, skip_threads_16951)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16951, local_tid_16935 -
                      squot32(local_tid_16935, 32) * 32) &&
                slt32(local_tid_16935, stage1_num_groups_16899)) {
                // write result
                {
                    ((volatile __local
                      double *) scan_arr_mem_16939)[local_tid_16935] = x_16283;
                    x_16285 = x_16283;
                    ((volatile __local
                      double *) scan_arr_mem_16941)[local_tid_16935] = x_16284;
                    x_16286 = x_16284;
                }
            }
            if (sle32(wave_sizze_16937, skip_threads_16951)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16951 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16935 - squot32(local_tid_16935, 32) * 32) == 31 &&
            slt32(local_tid_16935, stage1_num_groups_16899)) {
            ((volatile __local
              double *) scan_arr_mem_16939)[squot32(local_tid_16935, 32)] =
                x_16283;
            ((volatile __local
              double *) scan_arr_mem_16941)[squot32(local_tid_16935, 32)] =
                x_16284;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16953;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16935, 32) == 0 && slt32(local_tid_16935,
                                                           stage1_num_groups_16899)) {
                x_16946 = ((volatile __local
                            double *) scan_arr_mem_16939)[local_tid_16935];
                x_16947 = ((volatile __local
                            double *) scan_arr_mem_16941)[local_tid_16935];
                if ((local_tid_16935 - squot32(local_tid_16935, 32) * 32) ==
                    0) {
                    x_16944 = x_16946;
                    x_16945 = x_16947;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16953 = 1;
            while (slt32(skip_threads_16953, 32)) {
                if (sle32(skip_threads_16953, local_tid_16935 -
                          squot32(local_tid_16935, 32) * 32) &&
                    (squot32(local_tid_16935, 32) == 0 && slt32(local_tid_16935,
                                                                stage1_num_groups_16899))) {
                    // read operands
                    {
                        x_16944 = ((volatile __local
                                    double *) scan_arr_mem_16939)[local_tid_16935 -
                                                                  skip_threads_16953];
                        x_16945 = ((volatile __local
                                    double *) scan_arr_mem_16941)[local_tid_16935 -
                                                                  skip_threads_16953];
                    }
                    // perform operation
                    {
                        bool inactive_16954 = slt32(srem32((local_tid_16935 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_16278 *
                                                            sdiv_up32(n_13843 *
                                                                      m_13844,
                                                                      num_threads_16900)) -
                                                           1, m_13844),
                                                    (local_tid_16935 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_16278 *
                                                     sdiv_up32(n_13843 *
                                                               m_13844,
                                                               num_threads_16900)) -
                                                    1 - (((local_tid_16935 -
                                                           skip_threads_16953) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_16278 *
                                                          sdiv_up32(n_13843 *
                                                                    m_13844,
                                                                    num_threads_16900)) -
                                                         1));
                        
                        if (inactive_16954) {
                            x_16944 = x_16946;
                            x_16945 = x_16947;
                        }
                        if (!inactive_16954) {
                            double y_16948 = x_16944 * x_16947;
                            double res_16949 = x_16946 + y_16948;
                            double res_16950 = x_16945 * x_16947;
                            
                            x_16944 = res_16949;
                            x_16945 = res_16950;
                        }
                    }
                }
                if (sle32(wave_sizze_16937, skip_threads_16953)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16953, local_tid_16935 -
                          squot32(local_tid_16935, 32) * 32) &&
                    (squot32(local_tid_16935, 32) == 0 && slt32(local_tid_16935,
                                                                stage1_num_groups_16899))) {
                    // write result
                    {
                        ((volatile __local
                          double *) scan_arr_mem_16939)[local_tid_16935] =
                            x_16944;
                        x_16946 = x_16944;
                        ((volatile __local
                          double *) scan_arr_mem_16941)[local_tid_16935] =
                            x_16945;
                        x_16947 = x_16945;
                    }
                }
                if (sle32(wave_sizze_16937, skip_threads_16953)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16953 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16935, 32) == 0 || !slt32(local_tid_16935,
                                                          stage1_num_groups_16899))) {
            // read operands
            {
                x_16285 = x_16283;
                x_16286 = x_16284;
                x_16283 = ((__local
                            double *) scan_arr_mem_16939)[squot32(local_tid_16935,
                                                                  32) - 1];
                x_16284 = ((__local
                            double *) scan_arr_mem_16941)[squot32(local_tid_16935,
                                                                  32) - 1];
            }
            // perform operation
            {
                bool inactive_16955 = slt32(srem32((local_tid_16935 + 1) *
                                                   (segscan_group_sizze_16278 *
                                                    sdiv_up32(n_13843 * m_13844,
                                                              num_threads_16900)) -
                                                   1, m_13844),
                                            (local_tid_16935 + 1) *
                                            (segscan_group_sizze_16278 *
                                             sdiv_up32(n_13843 * m_13844,
                                                       num_threads_16900)) - 1 -
                                            ((squot32(local_tid_16935, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_16278 *
                                              sdiv_up32(n_13843 * m_13844,
                                                        num_threads_16900)) -
                                             1));
                
                if (inactive_16955) {
                    x_16283 = x_16285;
                    x_16284 = x_16286;
                }
                if (!inactive_16955) {
                    double y_16287 = x_16283 * x_16286;
                    double res_16288 = x_16285 + y_16287;
                    double res_16289 = x_16284 * x_16286;
                    
                    x_16283 = res_16288;
                    x_16284 = res_16289;
                }
            }
            // write final result
            {
                ((__local double *) scan_arr_mem_16939)[local_tid_16935] =
                    x_16283;
                ((__local double *) scan_arr_mem_16941)[local_tid_16935] =
                    x_16284;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16935, 32) == 0) {
            ((__local double *) scan_arr_mem_16939)[local_tid_16935] = x_16285;
            ((__local double *) scan_arr_mem_16941)[local_tid_16935] = x_16286;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_15531, n_13843) && slt32(gtid_15541, m_13844)) {
            ((__global double *) mem_16515)[gtid_15531 * m_13844 + gtid_15541] =
                ((__local double *) scan_arr_mem_16939)[local_tid_16935];
            ((__global double *) mem_16520)[gtid_15531 * m_13844 + gtid_15541] =
                ((__local double *) scan_arr_mem_16941)[local_tid_16935];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_16278
}
__kernel void tridagNestedziscan_stage2_15697(__global int *global_failure,
                                              __local volatile
                                              int64_t *scan_arr_mem_16858_backing_aligned_0,
                                              __local volatile
                                              int64_t *scan_arr_mem_16856_backing_aligned_1,
                                              int32_t n_13843, int32_t m_13844,
                                              __global unsigned char *mem_16494,
                                              __global unsigned char *mem_16499,
                                              int32_t stage1_num_groups_16816,
                                              int32_t num_threads_16817)
{
    #define segscan_group_sizze_16173 (tridagNestedzisegscan_group_sizze_15691)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16858_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16858_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16856_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16856_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16851;
    int32_t local_tid_16852;
    int32_t group_sizze_16855;
    int32_t wave_sizze_16854;
    int32_t group_tid_16853;
    
    global_tid_16851 = get_global_id(0);
    local_tid_16852 = get_local_id(0);
    group_sizze_16855 = get_local_size(0);
    wave_sizze_16854 = LOCKSTEP_WIDTH;
    group_tid_16853 = get_group_id(0);
    
    int32_t phys_tid_15697;
    
    phys_tid_15697 = global_tid_16851;
    
    __local char *scan_arr_mem_16856;
    __local char *scan_arr_mem_16858;
    
    scan_arr_mem_16856 = (__local char *) scan_arr_mem_16856_backing_0;
    scan_arr_mem_16858 = (__local char *) scan_arr_mem_16858_backing_1;
    
    int32_t flat_idx_16860;
    
    flat_idx_16860 = (local_tid_16852 + 1) * (segscan_group_sizze_16173 *
                                              sdiv_up32(n_13843 * m_13844,
                                                        num_threads_16817)) - 1;
    
    int32_t gtid_15686;
    
    gtid_15686 = squot32(flat_idx_16860, m_13844);
    
    int32_t gtid_15696;
    
    gtid_15696 = flat_idx_16860 - squot32(flat_idx_16860, m_13844) * m_13844;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_15686, n_13843) && slt32(gtid_15696, m_13844)) {
            ((__local double *) scan_arr_mem_16856)[local_tid_16852] =
                ((__global double *) mem_16494)[gtid_15686 * m_13844 +
                                                gtid_15696];
            ((__local double *) scan_arr_mem_16858)[local_tid_16852] =
                ((__global double *) mem_16499)[gtid_15686 * m_13844 +
                                                gtid_15696];
        } else {
            ((__local double *) scan_arr_mem_16856)[local_tid_16852] = 0.0;
            ((__local double *) scan_arr_mem_16858)[local_tid_16852] = 1.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    double x_16178;
    double x_16179;
    double x_16180;
    double x_16181;
    double x_16861;
    double x_16862;
    double x_16863;
    double x_16864;
    int32_t skip_threads_16868;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16852, stage1_num_groups_16816)) {
            x_16180 = ((volatile __local
                        double *) scan_arr_mem_16856)[local_tid_16852];
            x_16181 = ((volatile __local
                        double *) scan_arr_mem_16858)[local_tid_16852];
            if ((local_tid_16852 - squot32(local_tid_16852, 32) * 32) == 0) {
                x_16178 = x_16180;
                x_16179 = x_16181;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16868 = 1;
        while (slt32(skip_threads_16868, 32)) {
            if (sle32(skip_threads_16868, local_tid_16852 -
                      squot32(local_tid_16852, 32) * 32) &&
                slt32(local_tid_16852, stage1_num_groups_16816)) {
                // read operands
                {
                    x_16178 = ((volatile __local
                                double *) scan_arr_mem_16856)[local_tid_16852 -
                                                              skip_threads_16868];
                    x_16179 = ((volatile __local
                                double *) scan_arr_mem_16858)[local_tid_16852 -
                                                              skip_threads_16868];
                }
                // perform operation
                {
                    bool inactive_16869 = slt32(srem32((local_tid_16852 + 1) *
                                                       (segscan_group_sizze_16173 *
                                                        sdiv_up32(n_13843 *
                                                                  m_13844,
                                                                  num_threads_16817)) -
                                                       1, m_13844),
                                                (local_tid_16852 + 1) *
                                                (segscan_group_sizze_16173 *
                                                 sdiv_up32(n_13843 * m_13844,
                                                           num_threads_16817)) -
                                                1 - ((local_tid_16852 -
                                                      skip_threads_16868 + 1) *
                                                     (segscan_group_sizze_16173 *
                                                      sdiv_up32(n_13843 *
                                                                m_13844,
                                                                num_threads_16817)) -
                                                     1));
                    
                    if (inactive_16869) {
                        x_16178 = x_16180;
                        x_16179 = x_16181;
                    }
                    if (!inactive_16869) {
                        double y_16182 = x_16178 * x_16181;
                        double res_16183 = x_16180 + y_16182;
                        double res_16184 = x_16179 * x_16181;
                        
                        x_16178 = res_16183;
                        x_16179 = res_16184;
                    }
                }
            }
            if (sle32(wave_sizze_16854, skip_threads_16868)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16868, local_tid_16852 -
                      squot32(local_tid_16852, 32) * 32) &&
                slt32(local_tid_16852, stage1_num_groups_16816)) {
                // write result
                {
                    ((volatile __local
                      double *) scan_arr_mem_16856)[local_tid_16852] = x_16178;
                    x_16180 = x_16178;
                    ((volatile __local
                      double *) scan_arr_mem_16858)[local_tid_16852] = x_16179;
                    x_16181 = x_16179;
                }
            }
            if (sle32(wave_sizze_16854, skip_threads_16868)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16868 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16852 - squot32(local_tid_16852, 32) * 32) == 31 &&
            slt32(local_tid_16852, stage1_num_groups_16816)) {
            ((volatile __local
              double *) scan_arr_mem_16856)[squot32(local_tid_16852, 32)] =
                x_16178;
            ((volatile __local
              double *) scan_arr_mem_16858)[squot32(local_tid_16852, 32)] =
                x_16179;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16870;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16852, 32) == 0 && slt32(local_tid_16852,
                                                           stage1_num_groups_16816)) {
                x_16863 = ((volatile __local
                            double *) scan_arr_mem_16856)[local_tid_16852];
                x_16864 = ((volatile __local
                            double *) scan_arr_mem_16858)[local_tid_16852];
                if ((local_tid_16852 - squot32(local_tid_16852, 32) * 32) ==
                    0) {
                    x_16861 = x_16863;
                    x_16862 = x_16864;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16870 = 1;
            while (slt32(skip_threads_16870, 32)) {
                if (sle32(skip_threads_16870, local_tid_16852 -
                          squot32(local_tid_16852, 32) * 32) &&
                    (squot32(local_tid_16852, 32) == 0 && slt32(local_tid_16852,
                                                                stage1_num_groups_16816))) {
                    // read operands
                    {
                        x_16861 = ((volatile __local
                                    double *) scan_arr_mem_16856)[local_tid_16852 -
                                                                  skip_threads_16870];
                        x_16862 = ((volatile __local
                                    double *) scan_arr_mem_16858)[local_tid_16852 -
                                                                  skip_threads_16870];
                    }
                    // perform operation
                    {
                        bool inactive_16871 = slt32(srem32((local_tid_16852 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_16173 *
                                                            sdiv_up32(n_13843 *
                                                                      m_13844,
                                                                      num_threads_16817)) -
                                                           1, m_13844),
                                                    (local_tid_16852 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_16173 *
                                                     sdiv_up32(n_13843 *
                                                               m_13844,
                                                               num_threads_16817)) -
                                                    1 - (((local_tid_16852 -
                                                           skip_threads_16870) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_16173 *
                                                          sdiv_up32(n_13843 *
                                                                    m_13844,
                                                                    num_threads_16817)) -
                                                         1));
                        
                        if (inactive_16871) {
                            x_16861 = x_16863;
                            x_16862 = x_16864;
                        }
                        if (!inactive_16871) {
                            double y_16865 = x_16861 * x_16864;
                            double res_16866 = x_16863 + y_16865;
                            double res_16867 = x_16862 * x_16864;
                            
                            x_16861 = res_16866;
                            x_16862 = res_16867;
                        }
                    }
                }
                if (sle32(wave_sizze_16854, skip_threads_16870)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16870, local_tid_16852 -
                          squot32(local_tid_16852, 32) * 32) &&
                    (squot32(local_tid_16852, 32) == 0 && slt32(local_tid_16852,
                                                                stage1_num_groups_16816))) {
                    // write result
                    {
                        ((volatile __local
                          double *) scan_arr_mem_16856)[local_tid_16852] =
                            x_16861;
                        x_16863 = x_16861;
                        ((volatile __local
                          double *) scan_arr_mem_16858)[local_tid_16852] =
                            x_16862;
                        x_16864 = x_16862;
                    }
                }
                if (sle32(wave_sizze_16854, skip_threads_16870)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16870 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16852, 32) == 0 || !slt32(local_tid_16852,
                                                          stage1_num_groups_16816))) {
            // read operands
            {
                x_16180 = x_16178;
                x_16181 = x_16179;
                x_16178 = ((__local
                            double *) scan_arr_mem_16856)[squot32(local_tid_16852,
                                                                  32) - 1];
                x_16179 = ((__local
                            double *) scan_arr_mem_16858)[squot32(local_tid_16852,
                                                                  32) - 1];
            }
            // perform operation
            {
                bool inactive_16872 = slt32(srem32((local_tid_16852 + 1) *
                                                   (segscan_group_sizze_16173 *
                                                    sdiv_up32(n_13843 * m_13844,
                                                              num_threads_16817)) -
                                                   1, m_13844),
                                            (local_tid_16852 + 1) *
                                            (segscan_group_sizze_16173 *
                                             sdiv_up32(n_13843 * m_13844,
                                                       num_threads_16817)) - 1 -
                                            ((squot32(local_tid_16852, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_16173 *
                                              sdiv_up32(n_13843 * m_13844,
                                                        num_threads_16817)) -
                                             1));
                
                if (inactive_16872) {
                    x_16178 = x_16180;
                    x_16179 = x_16181;
                }
                if (!inactive_16872) {
                    double y_16182 = x_16178 * x_16181;
                    double res_16183 = x_16180 + y_16182;
                    double res_16184 = x_16179 * x_16181;
                    
                    x_16178 = res_16183;
                    x_16179 = res_16184;
                }
            }
            // write final result
            {
                ((__local double *) scan_arr_mem_16856)[local_tid_16852] =
                    x_16178;
                ((__local double *) scan_arr_mem_16858)[local_tid_16852] =
                    x_16179;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16852, 32) == 0) {
            ((__local double *) scan_arr_mem_16856)[local_tid_16852] = x_16180;
            ((__local double *) scan_arr_mem_16858)[local_tid_16852] = x_16181;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_15686, n_13843) && slt32(gtid_15696, m_13844)) {
            ((__global double *) mem_16494)[gtid_15686 * m_13844 + gtid_15696] =
                ((__local double *) scan_arr_mem_16856)[local_tid_16852];
            ((__global double *) mem_16499)[gtid_15686 * m_13844 + gtid_15696] =
                ((__local double *) scan_arr_mem_16858)[local_tid_16852];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_16173
}
__kernel void tridagNestedziscan_stage2_15930(__global int *global_failure,
                                              __local volatile
                                              int64_t *scan_arr_mem_16765_backing_aligned_0,
                                              __local volatile
                                              int64_t *scan_arr_mem_16763_backing_aligned_1,
                                              __local volatile
                                              int64_t *scan_arr_mem_16761_backing_aligned_2,
                                              __local volatile
                                              int64_t *scan_arr_mem_16759_backing_aligned_3,
                                              int32_t n_13843, int32_t m_13844,
                                              __global unsigned char *mem_16467,
                                              __global unsigned char *mem_16472,
                                              __global unsigned char *mem_16477,
                                              __global unsigned char *mem_16482,
                                              int32_t stage1_num_groups_16679,
                                              int32_t num_threads_16680)
{
    #define segscan_group_sizze_16005 (tridagNestedzisegscan_group_sizze_15924)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16765_backing_3 =
                          (__local volatile
                           char *) scan_arr_mem_16765_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16763_backing_2 =
                          (__local volatile
                           char *) scan_arr_mem_16763_backing_aligned_1;
    __local volatile char *restrict scan_arr_mem_16761_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16761_backing_aligned_2;
    __local volatile char *restrict scan_arr_mem_16759_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16759_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16754;
    int32_t local_tid_16755;
    int32_t group_sizze_16758;
    int32_t wave_sizze_16757;
    int32_t group_tid_16756;
    
    global_tid_16754 = get_global_id(0);
    local_tid_16755 = get_local_id(0);
    group_sizze_16758 = get_local_size(0);
    wave_sizze_16757 = LOCKSTEP_WIDTH;
    group_tid_16756 = get_group_id(0);
    
    int32_t phys_tid_15930;
    
    phys_tid_15930 = global_tid_16754;
    
    __local char *scan_arr_mem_16759;
    __local char *scan_arr_mem_16761;
    __local char *scan_arr_mem_16763;
    __local char *scan_arr_mem_16765;
    
    scan_arr_mem_16759 = (__local char *) scan_arr_mem_16759_backing_0;
    scan_arr_mem_16761 = (__local char *) scan_arr_mem_16761_backing_1;
    scan_arr_mem_16763 = (__local char *) scan_arr_mem_16763_backing_2;
    scan_arr_mem_16765 = (__local char *) scan_arr_mem_16765_backing_3;
    
    int32_t flat_idx_16767;
    
    flat_idx_16767 = (local_tid_16755 + 1) * (segscan_group_sizze_16005 *
                                              sdiv_up32(n_13843 * m_13844,
                                                        num_threads_16680)) - 1;
    
    int32_t gtid_15919;
    
    gtid_15919 = squot32(flat_idx_16767, m_13844);
    
    int32_t gtid_15929;
    
    gtid_15929 = flat_idx_16767 - squot32(flat_idx_16767, m_13844) * m_13844;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_15919, n_13843) && slt32(gtid_15929, m_13844)) {
            ((__local double *) scan_arr_mem_16759)[local_tid_16755] =
                ((__global double *) mem_16467)[gtid_15919 * m_13844 +
                                                gtid_15929];
            ((__local double *) scan_arr_mem_16761)[local_tid_16755] =
                ((__global double *) mem_16472)[gtid_15919 * m_13844 +
                                                gtid_15929];
            ((__local double *) scan_arr_mem_16763)[local_tid_16755] =
                ((__global double *) mem_16477)[gtid_15919 * m_13844 +
                                                gtid_15929];
            ((__local double *) scan_arr_mem_16765)[local_tid_16755] =
                ((__global double *) mem_16482)[gtid_15919 * m_13844 +
                                                gtid_15929];
        } else {
            ((__local double *) scan_arr_mem_16759)[local_tid_16755] = 1.0;
            ((__local double *) scan_arr_mem_16761)[local_tid_16755] = 0.0;
            ((__local double *) scan_arr_mem_16763)[local_tid_16755] = 0.0;
            ((__local double *) scan_arr_mem_16765)[local_tid_16755] = 1.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    double x_16012;
    double x_16013;
    double x_16014;
    double x_16015;
    double x_16016;
    double x_16017;
    double x_16018;
    double x_16019;
    double x_16768;
    double x_16769;
    double x_16770;
    double x_16771;
    double x_16772;
    double x_16773;
    double x_16774;
    double x_16775;
    int32_t skip_threads_16793;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16755, stage1_num_groups_16679)) {
            x_16016 = ((volatile __local
                        double *) scan_arr_mem_16759)[local_tid_16755];
            x_16017 = ((volatile __local
                        double *) scan_arr_mem_16761)[local_tid_16755];
            x_16018 = ((volatile __local
                        double *) scan_arr_mem_16763)[local_tid_16755];
            x_16019 = ((volatile __local
                        double *) scan_arr_mem_16765)[local_tid_16755];
            if ((local_tid_16755 - squot32(local_tid_16755, 32) * 32) == 0) {
                x_16012 = x_16016;
                x_16013 = x_16017;
                x_16014 = x_16018;
                x_16015 = x_16019;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16793 = 1;
        while (slt32(skip_threads_16793, 32)) {
            if (sle32(skip_threads_16793, local_tid_16755 -
                      squot32(local_tid_16755, 32) * 32) &&
                slt32(local_tid_16755, stage1_num_groups_16679)) {
                // read operands
                {
                    x_16012 = ((volatile __local
                                double *) scan_arr_mem_16759)[local_tid_16755 -
                                                              skip_threads_16793];
                    x_16013 = ((volatile __local
                                double *) scan_arr_mem_16761)[local_tid_16755 -
                                                              skip_threads_16793];
                    x_16014 = ((volatile __local
                                double *) scan_arr_mem_16763)[local_tid_16755 -
                                                              skip_threads_16793];
                    x_16015 = ((volatile __local
                                double *) scan_arr_mem_16765)[local_tid_16755 -
                                                              skip_threads_16793];
                }
                // perform operation
                {
                    bool inactive_16794 = slt32(srem32((local_tid_16755 + 1) *
                                                       (segscan_group_sizze_16005 *
                                                        sdiv_up32(n_13843 *
                                                                  m_13844,
                                                                  num_threads_16680)) -
                                                       1, m_13844),
                                                (local_tid_16755 + 1) *
                                                (segscan_group_sizze_16005 *
                                                 sdiv_up32(n_13843 * m_13844,
                                                           num_threads_16680)) -
                                                1 - ((local_tid_16755 -
                                                      skip_threads_16793 + 1) *
                                                     (segscan_group_sizze_16005 *
                                                      sdiv_up32(n_13843 *
                                                                m_13844,
                                                                num_threads_16680)) -
                                                     1));
                    
                    if (inactive_16794) {
                        x_16012 = x_16016;
                        x_16013 = x_16017;
                        x_16014 = x_16018;
                        x_16015 = x_16019;
                    }
                    if (!inactive_16794) {
                        double y_16020 = x_16012 * x_16016;
                        double value_16021 = 1.0 / y_16020;
                        double y_16022 = x_16014 * x_16017;
                        double x_16023 = y_16020 + y_16022;
                        double res_16024 = value_16021 * x_16023;
                        double x_16025 = x_16013 * x_16016;
                        double y_16026 = x_16015 * x_16017;
                        double x_16027 = x_16025 + y_16026;
                        double res_16028 = value_16021 * x_16027;
                        double x_16029 = x_16012 * x_16018;
                        double y_16030 = x_16014 * x_16019;
                        double x_16031 = x_16029 + y_16030;
                        double res_16032 = value_16021 * x_16031;
                        double x_16033 = x_16013 * x_16018;
                        double y_16034 = x_16015 * x_16019;
                        double x_16035 = x_16033 + y_16034;
                        double res_16036 = value_16021 * x_16035;
                        
                        x_16012 = res_16024;
                        x_16013 = res_16028;
                        x_16014 = res_16032;
                        x_16015 = res_16036;
                    }
                }
            }
            if (sle32(wave_sizze_16757, skip_threads_16793)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16793, local_tid_16755 -
                      squot32(local_tid_16755, 32) * 32) &&
                slt32(local_tid_16755, stage1_num_groups_16679)) {
                // write result
                {
                    ((volatile __local
                      double *) scan_arr_mem_16759)[local_tid_16755] = x_16012;
                    x_16016 = x_16012;
                    ((volatile __local
                      double *) scan_arr_mem_16761)[local_tid_16755] = x_16013;
                    x_16017 = x_16013;
                    ((volatile __local
                      double *) scan_arr_mem_16763)[local_tid_16755] = x_16014;
                    x_16018 = x_16014;
                    ((volatile __local
                      double *) scan_arr_mem_16765)[local_tid_16755] = x_16015;
                    x_16019 = x_16015;
                }
            }
            if (sle32(wave_sizze_16757, skip_threads_16793)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16793 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16755 - squot32(local_tid_16755, 32) * 32) == 31 &&
            slt32(local_tid_16755, stage1_num_groups_16679)) {
            ((volatile __local
              double *) scan_arr_mem_16759)[squot32(local_tid_16755, 32)] =
                x_16012;
            ((volatile __local
              double *) scan_arr_mem_16761)[squot32(local_tid_16755, 32)] =
                x_16013;
            ((volatile __local
              double *) scan_arr_mem_16763)[squot32(local_tid_16755, 32)] =
                x_16014;
            ((volatile __local
              double *) scan_arr_mem_16765)[squot32(local_tid_16755, 32)] =
                x_16015;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16795;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16755, 32) == 0 && slt32(local_tid_16755,
                                                           stage1_num_groups_16679)) {
                x_16772 = ((volatile __local
                            double *) scan_arr_mem_16759)[local_tid_16755];
                x_16773 = ((volatile __local
                            double *) scan_arr_mem_16761)[local_tid_16755];
                x_16774 = ((volatile __local
                            double *) scan_arr_mem_16763)[local_tid_16755];
                x_16775 = ((volatile __local
                            double *) scan_arr_mem_16765)[local_tid_16755];
                if ((local_tid_16755 - squot32(local_tid_16755, 32) * 32) ==
                    0) {
                    x_16768 = x_16772;
                    x_16769 = x_16773;
                    x_16770 = x_16774;
                    x_16771 = x_16775;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16795 = 1;
            while (slt32(skip_threads_16795, 32)) {
                if (sle32(skip_threads_16795, local_tid_16755 -
                          squot32(local_tid_16755, 32) * 32) &&
                    (squot32(local_tid_16755, 32) == 0 && slt32(local_tid_16755,
                                                                stage1_num_groups_16679))) {
                    // read operands
                    {
                        x_16768 = ((volatile __local
                                    double *) scan_arr_mem_16759)[local_tid_16755 -
                                                                  skip_threads_16795];
                        x_16769 = ((volatile __local
                                    double *) scan_arr_mem_16761)[local_tid_16755 -
                                                                  skip_threads_16795];
                        x_16770 = ((volatile __local
                                    double *) scan_arr_mem_16763)[local_tid_16755 -
                                                                  skip_threads_16795];
                        x_16771 = ((volatile __local
                                    double *) scan_arr_mem_16765)[local_tid_16755 -
                                                                  skip_threads_16795];
                    }
                    // perform operation
                    {
                        bool inactive_16796 = slt32(srem32((local_tid_16755 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_16005 *
                                                            sdiv_up32(n_13843 *
                                                                      m_13844,
                                                                      num_threads_16680)) -
                                                           1, m_13844),
                                                    (local_tid_16755 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_16005 *
                                                     sdiv_up32(n_13843 *
                                                               m_13844,
                                                               num_threads_16680)) -
                                                    1 - (((local_tid_16755 -
                                                           skip_threads_16795) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_16005 *
                                                          sdiv_up32(n_13843 *
                                                                    m_13844,
                                                                    num_threads_16680)) -
                                                         1));
                        
                        if (inactive_16796) {
                            x_16768 = x_16772;
                            x_16769 = x_16773;
                            x_16770 = x_16774;
                            x_16771 = x_16775;
                        }
                        if (!inactive_16796) {
                            double y_16776 = x_16768 * x_16772;
                            double value_16777 = 1.0 / y_16776;
                            double y_16778 = x_16770 * x_16773;
                            double x_16779 = y_16776 + y_16778;
                            double res_16780 = value_16777 * x_16779;
                            double x_16781 = x_16769 * x_16772;
                            double y_16782 = x_16771 * x_16773;
                            double x_16783 = x_16781 + y_16782;
                            double res_16784 = value_16777 * x_16783;
                            double x_16785 = x_16768 * x_16774;
                            double y_16786 = x_16770 * x_16775;
                            double x_16787 = x_16785 + y_16786;
                            double res_16788 = value_16777 * x_16787;
                            double x_16789 = x_16769 * x_16774;
                            double y_16790 = x_16771 * x_16775;
                            double x_16791 = x_16789 + y_16790;
                            double res_16792 = value_16777 * x_16791;
                            
                            x_16768 = res_16780;
                            x_16769 = res_16784;
                            x_16770 = res_16788;
                            x_16771 = res_16792;
                        }
                    }
                }
                if (sle32(wave_sizze_16757, skip_threads_16795)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16795, local_tid_16755 -
                          squot32(local_tid_16755, 32) * 32) &&
                    (squot32(local_tid_16755, 32) == 0 && slt32(local_tid_16755,
                                                                stage1_num_groups_16679))) {
                    // write result
                    {
                        ((volatile __local
                          double *) scan_arr_mem_16759)[local_tid_16755] =
                            x_16768;
                        x_16772 = x_16768;
                        ((volatile __local
                          double *) scan_arr_mem_16761)[local_tid_16755] =
                            x_16769;
                        x_16773 = x_16769;
                        ((volatile __local
                          double *) scan_arr_mem_16763)[local_tid_16755] =
                            x_16770;
                        x_16774 = x_16770;
                        ((volatile __local
                          double *) scan_arr_mem_16765)[local_tid_16755] =
                            x_16771;
                        x_16775 = x_16771;
                    }
                }
                if (sle32(wave_sizze_16757, skip_threads_16795)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16795 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16755, 32) == 0 || !slt32(local_tid_16755,
                                                          stage1_num_groups_16679))) {
            // read operands
            {
                x_16016 = x_16012;
                x_16017 = x_16013;
                x_16018 = x_16014;
                x_16019 = x_16015;
                x_16012 = ((__local
                            double *) scan_arr_mem_16759)[squot32(local_tid_16755,
                                                                  32) - 1];
                x_16013 = ((__local
                            double *) scan_arr_mem_16761)[squot32(local_tid_16755,
                                                                  32) - 1];
                x_16014 = ((__local
                            double *) scan_arr_mem_16763)[squot32(local_tid_16755,
                                                                  32) - 1];
                x_16015 = ((__local
                            double *) scan_arr_mem_16765)[squot32(local_tid_16755,
                                                                  32) - 1];
            }
            // perform operation
            {
                bool inactive_16797 = slt32(srem32((local_tid_16755 + 1) *
                                                   (segscan_group_sizze_16005 *
                                                    sdiv_up32(n_13843 * m_13844,
                                                              num_threads_16680)) -
                                                   1, m_13844),
                                            (local_tid_16755 + 1) *
                                            (segscan_group_sizze_16005 *
                                             sdiv_up32(n_13843 * m_13844,
                                                       num_threads_16680)) - 1 -
                                            ((squot32(local_tid_16755, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_16005 *
                                              sdiv_up32(n_13843 * m_13844,
                                                        num_threads_16680)) -
                                             1));
                
                if (inactive_16797) {
                    x_16012 = x_16016;
                    x_16013 = x_16017;
                    x_16014 = x_16018;
                    x_16015 = x_16019;
                }
                if (!inactive_16797) {
                    double y_16020 = x_16012 * x_16016;
                    double value_16021 = 1.0 / y_16020;
                    double y_16022 = x_16014 * x_16017;
                    double x_16023 = y_16020 + y_16022;
                    double res_16024 = value_16021 * x_16023;
                    double x_16025 = x_16013 * x_16016;
                    double y_16026 = x_16015 * x_16017;
                    double x_16027 = x_16025 + y_16026;
                    double res_16028 = value_16021 * x_16027;
                    double x_16029 = x_16012 * x_16018;
                    double y_16030 = x_16014 * x_16019;
                    double x_16031 = x_16029 + y_16030;
                    double res_16032 = value_16021 * x_16031;
                    double x_16033 = x_16013 * x_16018;
                    double y_16034 = x_16015 * x_16019;
                    double x_16035 = x_16033 + y_16034;
                    double res_16036 = value_16021 * x_16035;
                    
                    x_16012 = res_16024;
                    x_16013 = res_16028;
                    x_16014 = res_16032;
                    x_16015 = res_16036;
                }
            }
            // write final result
            {
                ((__local double *) scan_arr_mem_16759)[local_tid_16755] =
                    x_16012;
                ((__local double *) scan_arr_mem_16761)[local_tid_16755] =
                    x_16013;
                ((__local double *) scan_arr_mem_16763)[local_tid_16755] =
                    x_16014;
                ((__local double *) scan_arr_mem_16765)[local_tid_16755] =
                    x_16015;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16755, 32) == 0) {
            ((__local double *) scan_arr_mem_16759)[local_tid_16755] = x_16016;
            ((__local double *) scan_arr_mem_16761)[local_tid_16755] = x_16017;
            ((__local double *) scan_arr_mem_16763)[local_tid_16755] = x_16018;
            ((__local double *) scan_arr_mem_16765)[local_tid_16755] = x_16019;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_15919, n_13843) && slt32(gtid_15929, m_13844)) {
            ((__global double *) mem_16467)[gtid_15919 * m_13844 + gtid_15929] =
                ((__local double *) scan_arr_mem_16759)[local_tid_16755];
            ((__global double *) mem_16472)[gtid_15919 * m_13844 + gtid_15929] =
                ((__local double *) scan_arr_mem_16761)[local_tid_16755];
            ((__global double *) mem_16477)[gtid_15919 * m_13844 + gtid_15929] =
                ((__local double *) scan_arr_mem_16763)[local_tid_16755];
            ((__global double *) mem_16482)[gtid_15919 * m_13844 + gtid_15929] =
                ((__local double *) scan_arr_mem_16765)[local_tid_16755];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_16005
}
__kernel void tridagNestedziscan_stage3_15542(__global int *global_failure,
                                              int32_t n_13843, int32_t m_13844,
                                              int32_t num_groups_16279, __global
                                              unsigned char *mem_16515, __global
                                              unsigned char *mem_16520,
                                              int32_t num_threads_16900,
                                              int32_t required_groups_16956)
{
    #define segscan_group_sizze_16278 (tridagNestedzisegscan_group_sizze_15536)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16957;
    int32_t local_tid_16958;
    int32_t group_sizze_16961;
    int32_t wave_sizze_16960;
    int32_t group_tid_16959;
    
    global_tid_16957 = get_global_id(0);
    local_tid_16958 = get_local_id(0);
    group_sizze_16961 = get_local_size(0);
    wave_sizze_16960 = LOCKSTEP_WIDTH;
    group_tid_16959 = get_group_id(0);
    
    int32_t phys_tid_15542;
    
    phys_tid_15542 = global_tid_16957;
    
    int32_t phys_group_id_16962;
    
    phys_group_id_16962 = get_group_id(0);
    for (int32_t i_16963 = 0; i_16963 < sdiv_up32(required_groups_16956 -
                                                  phys_group_id_16962,
                                                  num_groups_16279);
         i_16963++) {
        int32_t virt_group_id_16964 = phys_group_id_16962 + i_16963 *
                num_groups_16279;
        int32_t flat_idx_16965 = virt_group_id_16964 *
                segscan_group_sizze_16278 + local_tid_16958;
        int32_t gtid_15531 = squot32(flat_idx_16965, m_13844);
        int32_t gtid_15541 = flat_idx_16965 - squot32(flat_idx_16965, m_13844) *
                m_13844;
        int32_t orig_group_16966 = squot32(flat_idx_16965,
                                           segscan_group_sizze_16278 *
                                           sdiv_up32(n_13843 * m_13844,
                                                     num_threads_16900));
        int32_t carry_in_flat_idx_16967 = orig_group_16966 *
                (segscan_group_sizze_16278 * sdiv_up32(n_13843 * m_13844,
                                                       num_threads_16900)) - 1;
        
        if (slt32(gtid_15531, n_13843) && slt32(gtid_15541, m_13844)) {
            if (!(orig_group_16966 == 0 || (flat_idx_16965 ==
                                            (orig_group_16966 + 1) *
                                            (segscan_group_sizze_16278 *
                                             sdiv_up32(n_13843 * m_13844,
                                                       num_threads_16900)) -
                                            1 || slt32(srem32(flat_idx_16965,
                                                              m_13844),
                                                       flat_idx_16965 -
                                                       carry_in_flat_idx_16967)))) {
                double x_16283;
                double x_16284;
                double x_16285;
                double x_16286;
                
                x_16283 = ((__global
                            double *) mem_16515)[squot32(carry_in_flat_idx_16967,
                                                         m_13844) * m_13844 +
                                                 (carry_in_flat_idx_16967 -
                                                  squot32(carry_in_flat_idx_16967,
                                                          m_13844) * m_13844)];
                x_16284 = ((__global
                            double *) mem_16520)[squot32(carry_in_flat_idx_16967,
                                                         m_13844) * m_13844 +
                                                 (carry_in_flat_idx_16967 -
                                                  squot32(carry_in_flat_idx_16967,
                                                          m_13844) * m_13844)];
                x_16285 = ((__global double *) mem_16515)[gtid_15531 * m_13844 +
                                                          gtid_15541];
                x_16286 = ((__global double *) mem_16520)[gtid_15531 * m_13844 +
                                                          gtid_15541];
                
                double y_16287;
                
                y_16287 = x_16283 * x_16286;
                
                double res_16288 = x_16285 + y_16287;
                double res_16289 = x_16284 * x_16286;
                
                x_16283 = res_16288;
                x_16284 = res_16289;
                ((__global double *) mem_16515)[gtid_15531 * m_13844 +
                                                gtid_15541] = x_16283;
                ((__global double *) mem_16520)[gtid_15531 * m_13844 +
                                                gtid_15541] = x_16284;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_16278
}
__kernel void tridagNestedziscan_stage3_15697(__global int *global_failure,
                                              int32_t n_13843, int32_t m_13844,
                                              int32_t num_groups_16174, __global
                                              unsigned char *mem_16494, __global
                                              unsigned char *mem_16499,
                                              int32_t num_threads_16817,
                                              int32_t required_groups_16873)
{
    #define segscan_group_sizze_16173 (tridagNestedzisegscan_group_sizze_15691)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16874;
    int32_t local_tid_16875;
    int32_t group_sizze_16878;
    int32_t wave_sizze_16877;
    int32_t group_tid_16876;
    
    global_tid_16874 = get_global_id(0);
    local_tid_16875 = get_local_id(0);
    group_sizze_16878 = get_local_size(0);
    wave_sizze_16877 = LOCKSTEP_WIDTH;
    group_tid_16876 = get_group_id(0);
    
    int32_t phys_tid_15697;
    
    phys_tid_15697 = global_tid_16874;
    
    int32_t phys_group_id_16879;
    
    phys_group_id_16879 = get_group_id(0);
    for (int32_t i_16880 = 0; i_16880 < sdiv_up32(required_groups_16873 -
                                                  phys_group_id_16879,
                                                  num_groups_16174);
         i_16880++) {
        int32_t virt_group_id_16881 = phys_group_id_16879 + i_16880 *
                num_groups_16174;
        int32_t flat_idx_16882 = virt_group_id_16881 *
                segscan_group_sizze_16173 + local_tid_16875;
        int32_t gtid_15686 = squot32(flat_idx_16882, m_13844);
        int32_t gtid_15696 = flat_idx_16882 - squot32(flat_idx_16882, m_13844) *
                m_13844;
        int32_t orig_group_16883 = squot32(flat_idx_16882,
                                           segscan_group_sizze_16173 *
                                           sdiv_up32(n_13843 * m_13844,
                                                     num_threads_16817));
        int32_t carry_in_flat_idx_16884 = orig_group_16883 *
                (segscan_group_sizze_16173 * sdiv_up32(n_13843 * m_13844,
                                                       num_threads_16817)) - 1;
        
        if (slt32(gtid_15686, n_13843) && slt32(gtid_15696, m_13844)) {
            if (!(orig_group_16883 == 0 || (flat_idx_16882 ==
                                            (orig_group_16883 + 1) *
                                            (segscan_group_sizze_16173 *
                                             sdiv_up32(n_13843 * m_13844,
                                                       num_threads_16817)) -
                                            1 || slt32(srem32(flat_idx_16882,
                                                              m_13844),
                                                       flat_idx_16882 -
                                                       carry_in_flat_idx_16884)))) {
                double x_16178;
                double x_16179;
                double x_16180;
                double x_16181;
                
                x_16178 = ((__global
                            double *) mem_16494)[squot32(carry_in_flat_idx_16884,
                                                         m_13844) * m_13844 +
                                                 (carry_in_flat_idx_16884 -
                                                  squot32(carry_in_flat_idx_16884,
                                                          m_13844) * m_13844)];
                x_16179 = ((__global
                            double *) mem_16499)[squot32(carry_in_flat_idx_16884,
                                                         m_13844) * m_13844 +
                                                 (carry_in_flat_idx_16884 -
                                                  squot32(carry_in_flat_idx_16884,
                                                          m_13844) * m_13844)];
                x_16180 = ((__global double *) mem_16494)[gtid_15686 * m_13844 +
                                                          gtid_15696];
                x_16181 = ((__global double *) mem_16499)[gtid_15686 * m_13844 +
                                                          gtid_15696];
                
                double y_16182;
                
                y_16182 = x_16178 * x_16181;
                
                double res_16183 = x_16180 + y_16182;
                double res_16184 = x_16179 * x_16181;
                
                x_16178 = res_16183;
                x_16179 = res_16184;
                ((__global double *) mem_16494)[gtid_15686 * m_13844 +
                                                gtid_15696] = x_16178;
                ((__global double *) mem_16499)[gtid_15686 * m_13844 +
                                                gtid_15696] = x_16179;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_16173
}
__kernel void tridagNestedziscan_stage3_15930(__global int *global_failure,
                                              int32_t n_13843, int32_t m_13844,
                                              int32_t num_groups_16006, __global
                                              unsigned char *mem_16467, __global
                                              unsigned char *mem_16472, __global
                                              unsigned char *mem_16477, __global
                                              unsigned char *mem_16482,
                                              int32_t num_threads_16680,
                                              int32_t required_groups_16798)
{
    #define segscan_group_sizze_16005 (tridagNestedzisegscan_group_sizze_15924)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16799;
    int32_t local_tid_16800;
    int32_t group_sizze_16803;
    int32_t wave_sizze_16802;
    int32_t group_tid_16801;
    
    global_tid_16799 = get_global_id(0);
    local_tid_16800 = get_local_id(0);
    group_sizze_16803 = get_local_size(0);
    wave_sizze_16802 = LOCKSTEP_WIDTH;
    group_tid_16801 = get_group_id(0);
    
    int32_t phys_tid_15930;
    
    phys_tid_15930 = global_tid_16799;
    
    int32_t phys_group_id_16804;
    
    phys_group_id_16804 = get_group_id(0);
    for (int32_t i_16805 = 0; i_16805 < sdiv_up32(required_groups_16798 -
                                                  phys_group_id_16804,
                                                  num_groups_16006);
         i_16805++) {
        int32_t virt_group_id_16806 = phys_group_id_16804 + i_16805 *
                num_groups_16006;
        int32_t flat_idx_16807 = virt_group_id_16806 *
                segscan_group_sizze_16005 + local_tid_16800;
        int32_t gtid_15919 = squot32(flat_idx_16807, m_13844);
        int32_t gtid_15929 = flat_idx_16807 - squot32(flat_idx_16807, m_13844) *
                m_13844;
        int32_t orig_group_16808 = squot32(flat_idx_16807,
                                           segscan_group_sizze_16005 *
                                           sdiv_up32(n_13843 * m_13844,
                                                     num_threads_16680));
        int32_t carry_in_flat_idx_16809 = orig_group_16808 *
                (segscan_group_sizze_16005 * sdiv_up32(n_13843 * m_13844,
                                                       num_threads_16680)) - 1;
        
        if (slt32(gtid_15919, n_13843) && slt32(gtid_15929, m_13844)) {
            if (!(orig_group_16808 == 0 || (flat_idx_16807 ==
                                            (orig_group_16808 + 1) *
                                            (segscan_group_sizze_16005 *
                                             sdiv_up32(n_13843 * m_13844,
                                                       num_threads_16680)) -
                                            1 || slt32(srem32(flat_idx_16807,
                                                              m_13844),
                                                       flat_idx_16807 -
                                                       carry_in_flat_idx_16809)))) {
                double x_16012;
                double x_16013;
                double x_16014;
                double x_16015;
                double x_16016;
                double x_16017;
                double x_16018;
                double x_16019;
                
                x_16012 = ((__global
                            double *) mem_16467)[squot32(carry_in_flat_idx_16809,
                                                         m_13844) * m_13844 +
                                                 (carry_in_flat_idx_16809 -
                                                  squot32(carry_in_flat_idx_16809,
                                                          m_13844) * m_13844)];
                x_16013 = ((__global
                            double *) mem_16472)[squot32(carry_in_flat_idx_16809,
                                                         m_13844) * m_13844 +
                                                 (carry_in_flat_idx_16809 -
                                                  squot32(carry_in_flat_idx_16809,
                                                          m_13844) * m_13844)];
                x_16014 = ((__global
                            double *) mem_16477)[squot32(carry_in_flat_idx_16809,
                                                         m_13844) * m_13844 +
                                                 (carry_in_flat_idx_16809 -
                                                  squot32(carry_in_flat_idx_16809,
                                                          m_13844) * m_13844)];
                x_16015 = ((__global
                            double *) mem_16482)[squot32(carry_in_flat_idx_16809,
                                                         m_13844) * m_13844 +
                                                 (carry_in_flat_idx_16809 -
                                                  squot32(carry_in_flat_idx_16809,
                                                          m_13844) * m_13844)];
                x_16016 = ((__global double *) mem_16467)[gtid_15919 * m_13844 +
                                                          gtid_15929];
                x_16017 = ((__global double *) mem_16472)[gtid_15919 * m_13844 +
                                                          gtid_15929];
                x_16018 = ((__global double *) mem_16477)[gtid_15919 * m_13844 +
                                                          gtid_15929];
                x_16019 = ((__global double *) mem_16482)[gtid_15919 * m_13844 +
                                                          gtid_15929];
                
                double y_16020;
                
                y_16020 = x_16012 * x_16016;
                
                double value_16021 = 1.0 / y_16020;
                double y_16022 = x_16014 * x_16017;
                double x_16023 = y_16020 + y_16022;
                double res_16024 = value_16021 * x_16023;
                double x_16025 = x_16013 * x_16016;
                double y_16026 = x_16015 * x_16017;
                double x_16027 = x_16025 + y_16026;
                double res_16028 = value_16021 * x_16027;
                double x_16029 = x_16012 * x_16018;
                double y_16030 = x_16014 * x_16019;
                double x_16031 = x_16029 + y_16030;
                double res_16032 = value_16021 * x_16031;
                double x_16033 = x_16013 * x_16018;
                double y_16034 = x_16015 * x_16019;
                double x_16035 = x_16033 + y_16034;
                double res_16036 = value_16021 * x_16035;
                
                x_16012 = res_16024;
                x_16013 = res_16028;
                x_16014 = res_16032;
                x_16015 = res_16036;
                ((__global double *) mem_16467)[gtid_15919 * m_13844 +
                                                gtid_15929] = x_16012;
                ((__global double *) mem_16472)[gtid_15919 * m_13844 +
                                                gtid_15929] = x_16013;
                ((__global double *) mem_16477)[gtid_15919 * m_13844 +
                                                gtid_15929] = x_16014;
                ((__global double *) mem_16482)[gtid_15919 * m_13844 +
                                                gtid_15929] = x_16015;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_16005
}
__kernel void tridagNestedzisegmap_15399(__global int *global_failure,
                                         int32_t n_13843, int32_t m_13844,
                                         __global unsigned char *mem_16526,
                                         __global unsigned char *mem_16532)
{
    #define segmap_group_sizze_16380 (tridagNestedzisegmap_group_sizze_15404)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16973;
    int32_t local_tid_16974;
    int32_t group_sizze_16977;
    int32_t wave_sizze_16976;
    int32_t group_tid_16975;
    
    global_tid_16973 = get_global_id(0);
    local_tid_16974 = get_local_id(0);
    group_sizze_16977 = get_local_size(0);
    wave_sizze_16976 = LOCKSTEP_WIDTH;
    group_tid_16975 = get_group_id(0);
    
    int32_t phys_tid_15399;
    
    phys_tid_15399 = global_tid_16973;
    
    int32_t gtid_15397;
    
    gtid_15397 = squot32(group_tid_16975 * segmap_group_sizze_16380 +
                         local_tid_16974, m_13844);
    
    int32_t gtid_15398;
    
    gtid_15398 = group_tid_16975 * segmap_group_sizze_16380 + local_tid_16974 -
        squot32(group_tid_16975 * segmap_group_sizze_16380 + local_tid_16974,
                m_13844) * m_13844;
    if (slt32(gtid_15397, n_13843) && slt32(gtid_15398, m_13844)) {
        int32_t x_16387 = sub32(m_13844, gtid_15398);
        int32_t i_16388 = sub32(x_16387, 1);
        double res_16389 = ((__global double *) mem_16526)[gtid_15397 *
                                                           m_13844 + i_16388];
        
        ((__global double *) mem_16532)[gtid_15397 * m_13844 + gtid_15398] =
            res_16389;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_16380
}
__kernel void tridagNestedzisegmap_15473(__global int *global_failure,
                                         int32_t n_13843, int32_t m_13844,
                                         __global unsigned char *mem_16509,
                                         __global unsigned char *mem_16515,
                                         __global unsigned char *mem_16520,
                                         __global unsigned char *mem_16526)
{
    #define segmap_group_sizze_16340 (tridagNestedzisegmap_group_sizze_15478)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16968;
    int32_t local_tid_16969;
    int32_t group_sizze_16972;
    int32_t wave_sizze_16971;
    int32_t group_tid_16970;
    
    global_tid_16968 = get_global_id(0);
    local_tid_16969 = get_local_id(0);
    group_sizze_16972 = get_local_size(0);
    wave_sizze_16971 = LOCKSTEP_WIDTH;
    group_tid_16970 = get_group_id(0);
    
    int32_t phys_tid_15473;
    
    phys_tid_15473 = global_tid_16968;
    
    int32_t gtid_15471;
    
    gtid_15471 = squot32(group_tid_16970 * segmap_group_sizze_16340 +
                         local_tid_16969, m_13844);
    
    int32_t gtid_15472;
    
    gtid_15472 = group_tid_16970 * segmap_group_sizze_16340 + local_tid_16969 -
        squot32(group_tid_16970 * segmap_group_sizze_16340 + local_tid_16969,
                m_13844) * m_13844;
    if (slt32(gtid_15471, n_13843) && slt32(gtid_15472, m_13844)) {
        double yn_16345 = ((__global double *) mem_16509)[gtid_15471];
        double x_16346 = ((__global double *) mem_16515)[gtid_15471 * m_13844 +
                                                         gtid_15472];
        double x_16347 = ((__global double *) mem_16520)[gtid_15471 * m_13844 +
                                                         gtid_15472];
        double y_16351 = yn_16345 * x_16347;
        double res_16352 = x_16346 + y_16351;
        
        ((__global double *) mem_16526)[gtid_15471 * m_13844 + gtid_15472] =
            res_16352;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_16340
}
__kernel void tridagNestedzisegmap_15567(__global int *global_failure,
                                         int32_t n_13843, int32_t m_13844,
                                         int32_t i_13871,
                                         int32_t num_groups_16261, __global
                                         unsigned char *mem_16488, __global
                                         unsigned char *mem_16505, __global
                                         unsigned char *mem_16509)
{
    #define segmap_group_sizze_16260 (tridagNestedzisegmap_group_sizze_15570)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16890;
    int32_t local_tid_16891;
    int32_t group_sizze_16894;
    int32_t wave_sizze_16893;
    int32_t group_tid_16892;
    
    global_tid_16890 = get_global_id(0);
    local_tid_16891 = get_local_id(0);
    group_sizze_16894 = get_local_size(0);
    wave_sizze_16893 = LOCKSTEP_WIDTH;
    group_tid_16892 = get_group_id(0);
    
    int32_t phys_tid_15567;
    
    phys_tid_15567 = global_tid_16890;
    
    int32_t phys_group_id_16895;
    
    phys_group_id_16895 = get_group_id(0);
    for (int32_t i_16896 = 0; i_16896 < sdiv_up32(sdiv_up32(n_13843,
                                                            segmap_group_sizze_16260) -
                                                  phys_group_id_16895,
                                                  num_groups_16261);
         i_16896++) {
        int32_t virt_group_id_16897 = phys_group_id_16895 + i_16896 *
                num_groups_16261;
        int32_t gtid_15566 = virt_group_id_16897 * segmap_group_sizze_16260 +
                local_tid_16891;
        
        if (slt32(gtid_15566, n_13843)) {
            double x_16267 = ((__global double *) mem_16505)[gtid_15566 *
                                                             m_13844 + i_13871];
            double y_16268 = ((__global double *) mem_16488)[gtid_15566 *
                                                             m_13844 + i_13871];
            double yn_16269 = x_16267 / y_16268;
            
            ((__global double *) mem_16509)[gtid_15566] = yn_16269;
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_16260
}
__kernel void tridagNestedzisegmap_15628(__global int *global_failure,
                                         int32_t n_13843, int32_t m_13844,
                                         int32_t m_13850, __global
                                         unsigned char *y_mem_16412, __global
                                         unsigned char *mem_16494, __global
                                         unsigned char *mem_16499, __global
                                         unsigned char *mem_16505)
{
    #define segmap_group_sizze_16233 (tridagNestedzisegmap_group_sizze_15633)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16885;
    int32_t local_tid_16886;
    int32_t group_sizze_16889;
    int32_t wave_sizze_16888;
    int32_t group_tid_16887;
    
    global_tid_16885 = get_global_id(0);
    local_tid_16886 = get_local_id(0);
    group_sizze_16889 = get_local_size(0);
    wave_sizze_16888 = LOCKSTEP_WIDTH;
    group_tid_16887 = get_group_id(0);
    
    int32_t phys_tid_15628;
    
    phys_tid_15628 = global_tid_16885;
    
    int32_t gtid_15626;
    
    gtid_15626 = squot32(group_tid_16887 * segmap_group_sizze_16233 +
                         local_tid_16886, m_13844);
    
    int32_t gtid_15627;
    
    gtid_15627 = group_tid_16887 * segmap_group_sizze_16233 + local_tid_16886 -
        squot32(group_tid_16887 * segmap_group_sizze_16233 + local_tid_16886,
                m_13844) * m_13844;
    if (slt32(gtid_15626, n_13843) && slt32(gtid_15627, m_13844)) {
        double as_transformed_row_16238 = ((__global
                                            double *) y_mem_16412)[gtid_15626 *
                                                                   m_13850];
        double x_16239 = ((__global double *) mem_16494)[gtid_15626 * m_13844 +
                                                         gtid_15627];
        double x_16240 = ((__global double *) mem_16499)[gtid_15626 * m_13844 +
                                                         gtid_15627];
        double y_16244 = as_transformed_row_16238 * x_16240;
        double res_16245 = x_16239 + y_16244;
        
        ((__global double *) mem_16505)[gtid_15626 * m_13844 + gtid_15627] =
            res_16245;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_16233
}
__kernel void tridagNestedzisegmap_15803(__global int *global_failure,
                                         int32_t n_13843, int32_t m_13844,
                                         int32_t m_13846, __global
                                         unsigned char *b_mem_16410, __global
                                         unsigned char *mem_16467, __global
                                         unsigned char *mem_16472, __global
                                         unsigned char *mem_16477, __global
                                         unsigned char *mem_16482, __global
                                         unsigned char *mem_16488)
{
    #define segmap_group_sizze_16109 (tridagNestedzisegmap_group_sizze_15808)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16810;
    int32_t local_tid_16811;
    int32_t group_sizze_16814;
    int32_t wave_sizze_16813;
    int32_t group_tid_16812;
    
    global_tid_16810 = get_global_id(0);
    local_tid_16811 = get_local_id(0);
    group_sizze_16814 = get_local_size(0);
    wave_sizze_16813 = LOCKSTEP_WIDTH;
    group_tid_16812 = get_group_id(0);
    
    int32_t phys_tid_15803;
    
    phys_tid_15803 = global_tid_16810;
    
    int32_t gtid_15801;
    
    gtid_15801 = squot32(group_tid_16812 * segmap_group_sizze_16109 +
                         local_tid_16811, m_13844);
    
    int32_t gtid_15802;
    
    gtid_15802 = group_tid_16812 * segmap_group_sizze_16109 + local_tid_16811 -
        squot32(group_tid_16812 * segmap_group_sizze_16109 + local_tid_16811,
                m_13844) * m_13844;
    if (slt32(gtid_15801, n_13843) && slt32(gtid_15802, m_13844)) {
        double as_transformed_row_16114 = ((__global
                                            double *) b_mem_16410)[gtid_15801 *
                                                                   m_13846];
        double x_16115 = ((__global double *) mem_16467)[gtid_15801 * m_13844 +
                                                         gtid_15802];
        double x_16116 = ((__global double *) mem_16472)[gtid_15801 * m_13844 +
                                                         gtid_15802];
        double x_16117 = ((__global double *) mem_16477)[gtid_15801 * m_13844 +
                                                         gtid_15802];
        double x_16118 = ((__global double *) mem_16482)[gtid_15801 * m_13844 +
                                                         gtid_15802];
        double value_16120 = 1.0 / x_16115;
        double res_16123 = x_16115 * value_16120;
        double res_16127 = x_16116 * value_16120;
        double res_16131 = x_16117 * value_16120;
        double res_16135 = x_16118 * value_16120;
        double x_16136 = as_transformed_row_16114 * res_16123;
        double x_16137 = res_16127 + x_16136;
        double x_16138 = as_transformed_row_16114 * res_16131;
        double y_16139 = res_16135 + x_16138;
        double res_16140 = x_16137 / y_16139;
        
        ((__global double *) mem_16488)[gtid_15801 * m_13844 + gtid_15802] =
            res_16140;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_16109
}
__kernel void tridagNestedzisegmap_intragroup_14717(__global
                                                    int *global_failure,
                                                    __local volatile
                                                    int64_t *mem_16456_backing_aligned_0,
                                                    __local volatile
                                                    int64_t *mem_16452_backing_aligned_1,
                                                    __local volatile
                                                    int64_t *mem_16448_backing_aligned_2,
                                                    __local volatile
                                                    int64_t *mem_16445_backing_aligned_3,
                                                    __local volatile
                                                    int64_t *mem_16441_backing_aligned_4,
                                                    __local volatile
                                                    int64_t *mem_16437_backing_aligned_5,
                                                    __local volatile
                                                    int64_t *mem_16434_backing_aligned_6,
                                                    __local volatile
                                                    int64_t *mem_16430_backing_aligned_7,
                                                    __local volatile
                                                    int64_t *mem_16426_backing_aligned_8,
                                                    __local volatile
                                                    int64_t *mem_16423_backing_aligned_9,
                                                    __local volatile
                                                    int64_t *mem_16420_backing_aligned_10,
                                                    __local volatile
                                                    int64_t *mem_16417_backing_aligned_11,
                                                    int32_t m_13844,
                                                    int32_t m_13846,
                                                    int32_t m_13848,
                                                    int32_t m_13850,
                                                    int32_t i_13871, __global
                                                    unsigned char *a_mem_16409,
                                                    __global
                                                    unsigned char *b_mem_16410,
                                                    __global
                                                    unsigned char *c_mem_16411,
                                                    __global
                                                    unsigned char *y_mem_16412,
                                                    __global
                                                    unsigned char *mem_16461)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_16456_backing_11 = (__local volatile
                                                            char *) mem_16456_backing_aligned_0;
    __local volatile char *restrict mem_16452_backing_10 = (__local volatile
                                                            char *) mem_16452_backing_aligned_1;
    __local volatile char *restrict mem_16448_backing_9 = (__local volatile
                                                           char *) mem_16448_backing_aligned_2;
    __local volatile char *restrict mem_16445_backing_8 = (__local volatile
                                                           char *) mem_16445_backing_aligned_3;
    __local volatile char *restrict mem_16441_backing_7 = (__local volatile
                                                           char *) mem_16441_backing_aligned_4;
    __local volatile char *restrict mem_16437_backing_6 = (__local volatile
                                                           char *) mem_16437_backing_aligned_5;
    __local volatile char *restrict mem_16434_backing_5 = (__local volatile
                                                           char *) mem_16434_backing_aligned_6;
    __local volatile char *restrict mem_16430_backing_4 = (__local volatile
                                                           char *) mem_16430_backing_aligned_7;
    __local volatile char *restrict mem_16426_backing_3 = (__local volatile
                                                           char *) mem_16426_backing_aligned_8;
    __local volatile char *restrict mem_16423_backing_2 = (__local volatile
                                                           char *) mem_16423_backing_aligned_9;
    __local volatile char *restrict mem_16420_backing_1 = (__local volatile
                                                           char *) mem_16420_backing_aligned_10;
    __local volatile char *restrict mem_16417_backing_0 = (__local volatile
                                                           char *) mem_16417_backing_aligned_11;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16607;
    int32_t local_tid_16608;
    int32_t group_sizze_16611;
    int32_t wave_sizze_16610;
    int32_t group_tid_16609;
    
    global_tid_16607 = get_global_id(0);
    local_tid_16608 = get_local_id(0);
    group_sizze_16611 = get_local_size(0);
    wave_sizze_16610 = LOCKSTEP_WIDTH;
    group_tid_16609 = get_group_id(0);
    
    int32_t phys_tid_14717;
    
    phys_tid_14717 = group_tid_16609;
    
    int32_t ltid_pre_16612;
    
    ltid_pre_16612 = local_tid_16608;
    
    int32_t gtid_14656;
    
    gtid_14656 = group_tid_16609;
    
    double as_transformed_row_15167;
    
    as_transformed_row_15167 = ((__global double *) b_mem_16410)[gtid_14656 *
                                                                 m_13846];
    
    double as_transformed_row_15168 = ((__global
                                        double *) y_mem_16412)[gtid_14656 *
                                                               m_13850];
    __local char *mem_16417;
    
    mem_16417 = (__local char *) mem_16417_backing_0;
    
    __local char *mem_16420;
    
    mem_16420 = (__local char *) mem_16420_backing_1;
    
    __local char *mem_16423;
    
    mem_16423 = (__local char *) mem_16423_backing_2;
    
    __local char *mem_16426;
    
    mem_16426 = (__local char *) mem_16426_backing_3;
    
    int32_t gtid_14659 = ltid_pre_16612;
    int32_t phys_tid_14660 = local_tid_16608;
    
    if (slt32(gtid_14659, m_13844)) {
        bool cond_15209 = slt32(0, gtid_14659);
        double res_15210;
        
        if (cond_15209) {
            res_15210 = 1.0;
        } else {
            res_15210 = 0.0;
        }
        
        double res_15211;
        
        if (cond_15209) {
            res_15211 = 0.0;
        } else {
            res_15211 = 1.0;
        }
        
        double res_15212;
        
        if (cond_15209) {
            double x_elem_15207 = ((__global double *) b_mem_16410)[gtid_14656 *
                                                                    m_13846 +
                                                                    gtid_14659];
            
            res_15212 = x_elem_15207;
        } else {
            res_15212 = 1.0;
        }
        
        double res_15213;
        
        if (cond_15209) {
            double x_elem_15208 = ((__global double *) a_mem_16409)[gtid_14656 *
                                                                    m_13844 +
                                                                    gtid_14659];
            int32_t i_15214 = sub32(gtid_14659, 1);
            double y_15215 = ((__global double *) c_mem_16411)[gtid_14656 *
                                                               m_13848 +
                                                               i_15214];
            double y_15216 = x_elem_15208 * y_15215;
            double res_15217 = 0.0 - y_15216;
            
            res_15213 = res_15217;
        } else {
            res_15213 = 0.0;
        }
        ((__local double *) mem_16417)[gtid_14659] = res_15212;
        ((__local double *) mem_16420)[gtid_14659] = res_15213;
        ((__local double *) mem_16423)[gtid_14659] = res_15210;
        ((__local double *) mem_16426)[gtid_14659] = res_15211;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_16613;
    
    dims_flat_16613 = m_13844;
    
    double x_15181;
    double x_15182;
    double x_15183;
    double x_15184;
    double x_15185;
    double x_15186;
    double x_15187;
    double x_15188;
    double x_16618;
    double x_16619;
    double x_16620;
    double x_16621;
    double x_16622;
    double x_16623;
    double x_16624;
    double x_16625;
    int32_t skip_threads_16643;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16608, m_13844)) {
            x_15185 = ((volatile __local double *) mem_16417)[local_tid_16608];
            x_15186 = ((volatile __local double *) mem_16420)[local_tid_16608];
            x_15187 = ((volatile __local double *) mem_16423)[local_tid_16608];
            x_15188 = ((volatile __local double *) mem_16426)[local_tid_16608];
            if ((local_tid_16608 - squot32(local_tid_16608, 32) * 32) == 0) {
                x_15181 = x_15185;
                x_15182 = x_15186;
                x_15183 = x_15187;
                x_15184 = x_15188;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16643 = 1;
        while (slt32(skip_threads_16643, 32)) {
            if (sle32(skip_threads_16643, local_tid_16608 -
                      squot32(local_tid_16608, 32) * 32) &&
                slt32(local_tid_16608, m_13844)) {
                // read operands
                {
                    x_15181 = ((volatile __local
                                double *) mem_16417)[local_tid_16608 -
                                                     skip_threads_16643];
                    x_15182 = ((volatile __local
                                double *) mem_16420)[local_tid_16608 -
                                                     skip_threads_16643];
                    x_15183 = ((volatile __local
                                double *) mem_16423)[local_tid_16608 -
                                                     skip_threads_16643];
                    x_15184 = ((volatile __local
                                double *) mem_16426)[local_tid_16608 -
                                                     skip_threads_16643];
                }
                // perform operation
                {
                    bool inactive_16644 = slt32(srem32(local_tid_16608,
                                                       m_13844),
                                                local_tid_16608 -
                                                (local_tid_16608 -
                                                 skip_threads_16643));
                    
                    if (inactive_16644) {
                        x_15181 = x_15185;
                        x_15182 = x_15186;
                        x_15183 = x_15187;
                        x_15184 = x_15188;
                    }
                    if (!inactive_16644) {
                        double y_15189 = x_15181 * x_15185;
                        double value_15190 = 1.0 / y_15189;
                        double y_15191 = x_15183 * x_15186;
                        double x_15192 = y_15189 + y_15191;
                        double res_15193 = value_15190 * x_15192;
                        double x_15194 = x_15182 * x_15185;
                        double y_15195 = x_15184 * x_15186;
                        double x_15196 = x_15194 + y_15195;
                        double res_15197 = value_15190 * x_15196;
                        double x_15198 = x_15181 * x_15187;
                        double y_15199 = x_15183 * x_15188;
                        double x_15200 = x_15198 + y_15199;
                        double res_15201 = value_15190 * x_15200;
                        double x_15202 = x_15182 * x_15187;
                        double y_15203 = x_15184 * x_15188;
                        double x_15204 = x_15202 + y_15203;
                        double res_15205 = value_15190 * x_15204;
                        
                        x_15181 = res_15193;
                        x_15182 = res_15197;
                        x_15183 = res_15201;
                        x_15184 = res_15205;
                    }
                }
            }
            if (sle32(wave_sizze_16610, skip_threads_16643)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16643, local_tid_16608 -
                      squot32(local_tid_16608, 32) * 32) &&
                slt32(local_tid_16608, m_13844)) {
                // write result
                {
                    ((volatile __local double *) mem_16417)[local_tid_16608] =
                        x_15181;
                    x_15185 = x_15181;
                    ((volatile __local double *) mem_16420)[local_tid_16608] =
                        x_15182;
                    x_15186 = x_15182;
                    ((volatile __local double *) mem_16423)[local_tid_16608] =
                        x_15183;
                    x_15187 = x_15183;
                    ((volatile __local double *) mem_16426)[local_tid_16608] =
                        x_15184;
                    x_15188 = x_15184;
                }
            }
            if (sle32(wave_sizze_16610, skip_threads_16643)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16643 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16608 - squot32(local_tid_16608, 32) * 32) == 31 &&
            slt32(local_tid_16608, m_13844)) {
            ((volatile __local double *) mem_16417)[squot32(local_tid_16608,
                                                            32)] = x_15181;
            ((volatile __local double *) mem_16420)[squot32(local_tid_16608,
                                                            32)] = x_15182;
            ((volatile __local double *) mem_16423)[squot32(local_tid_16608,
                                                            32)] = x_15183;
            ((volatile __local double *) mem_16426)[squot32(local_tid_16608,
                                                            32)] = x_15184;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16645;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16608, 32) == 0 && slt32(local_tid_16608,
                                                           m_13844)) {
                x_16622 = ((volatile __local
                            double *) mem_16417)[local_tid_16608];
                x_16623 = ((volatile __local
                            double *) mem_16420)[local_tid_16608];
                x_16624 = ((volatile __local
                            double *) mem_16423)[local_tid_16608];
                x_16625 = ((volatile __local
                            double *) mem_16426)[local_tid_16608];
                if ((local_tid_16608 - squot32(local_tid_16608, 32) * 32) ==
                    0) {
                    x_16618 = x_16622;
                    x_16619 = x_16623;
                    x_16620 = x_16624;
                    x_16621 = x_16625;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16645 = 1;
            while (slt32(skip_threads_16645, 32)) {
                if (sle32(skip_threads_16645, local_tid_16608 -
                          squot32(local_tid_16608, 32) * 32) &&
                    (squot32(local_tid_16608, 32) == 0 && slt32(local_tid_16608,
                                                                m_13844))) {
                    // read operands
                    {
                        x_16618 = ((volatile __local
                                    double *) mem_16417)[local_tid_16608 -
                                                         skip_threads_16645];
                        x_16619 = ((volatile __local
                                    double *) mem_16420)[local_tid_16608 -
                                                         skip_threads_16645];
                        x_16620 = ((volatile __local
                                    double *) mem_16423)[local_tid_16608 -
                                                         skip_threads_16645];
                        x_16621 = ((volatile __local
                                    double *) mem_16426)[local_tid_16608 -
                                                         skip_threads_16645];
                    }
                    // perform operation
                    {
                        bool inactive_16646 = slt32(srem32(local_tid_16608 *
                                                           32 + 32 - 1,
                                                           m_13844),
                                                    local_tid_16608 * 32 + 32 -
                                                    1 - ((local_tid_16608 -
                                                          skip_threads_16645) *
                                                         32 + 32 - 1));
                        
                        if (inactive_16646) {
                            x_16618 = x_16622;
                            x_16619 = x_16623;
                            x_16620 = x_16624;
                            x_16621 = x_16625;
                        }
                        if (!inactive_16646) {
                            double y_16626 = x_16618 * x_16622;
                            double value_16627 = 1.0 / y_16626;
                            double y_16628 = x_16620 * x_16623;
                            double x_16629 = y_16626 + y_16628;
                            double res_16630 = value_16627 * x_16629;
                            double x_16631 = x_16619 * x_16622;
                            double y_16632 = x_16621 * x_16623;
                            double x_16633 = x_16631 + y_16632;
                            double res_16634 = value_16627 * x_16633;
                            double x_16635 = x_16618 * x_16624;
                            double y_16636 = x_16620 * x_16625;
                            double x_16637 = x_16635 + y_16636;
                            double res_16638 = value_16627 * x_16637;
                            double x_16639 = x_16619 * x_16624;
                            double y_16640 = x_16621 * x_16625;
                            double x_16641 = x_16639 + y_16640;
                            double res_16642 = value_16627 * x_16641;
                            
                            x_16618 = res_16630;
                            x_16619 = res_16634;
                            x_16620 = res_16638;
                            x_16621 = res_16642;
                        }
                    }
                }
                if (sle32(wave_sizze_16610, skip_threads_16645)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16645, local_tid_16608 -
                          squot32(local_tid_16608, 32) * 32) &&
                    (squot32(local_tid_16608, 32) == 0 && slt32(local_tid_16608,
                                                                m_13844))) {
                    // write result
                    {
                        ((volatile __local
                          double *) mem_16417)[local_tid_16608] = x_16618;
                        x_16622 = x_16618;
                        ((volatile __local
                          double *) mem_16420)[local_tid_16608] = x_16619;
                        x_16623 = x_16619;
                        ((volatile __local
                          double *) mem_16423)[local_tid_16608] = x_16620;
                        x_16624 = x_16620;
                        ((volatile __local
                          double *) mem_16426)[local_tid_16608] = x_16621;
                        x_16625 = x_16621;
                    }
                }
                if (sle32(wave_sizze_16610, skip_threads_16645)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16645 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16608, 32) == 0 || !slt32(local_tid_16608,
                                                          m_13844))) {
            // read operands
            {
                x_15185 = x_15181;
                x_15186 = x_15182;
                x_15187 = x_15183;
                x_15188 = x_15184;
                x_15181 = ((__local
                            double *) mem_16417)[squot32(local_tid_16608, 32) -
                                                 1];
                x_15182 = ((__local
                            double *) mem_16420)[squot32(local_tid_16608, 32) -
                                                 1];
                x_15183 = ((__local
                            double *) mem_16423)[squot32(local_tid_16608, 32) -
                                                 1];
                x_15184 = ((__local
                            double *) mem_16426)[squot32(local_tid_16608, 32) -
                                                 1];
            }
            // perform operation
            {
                bool inactive_16647 = slt32(srem32(local_tid_16608, m_13844),
                                            local_tid_16608 -
                                            (squot32(local_tid_16608, 32) * 32 -
                                             1));
                
                if (inactive_16647) {
                    x_15181 = x_15185;
                    x_15182 = x_15186;
                    x_15183 = x_15187;
                    x_15184 = x_15188;
                }
                if (!inactive_16647) {
                    double y_15189 = x_15181 * x_15185;
                    double value_15190 = 1.0 / y_15189;
                    double y_15191 = x_15183 * x_15186;
                    double x_15192 = y_15189 + y_15191;
                    double res_15193 = value_15190 * x_15192;
                    double x_15194 = x_15182 * x_15185;
                    double y_15195 = x_15184 * x_15186;
                    double x_15196 = x_15194 + y_15195;
                    double res_15197 = value_15190 * x_15196;
                    double x_15198 = x_15181 * x_15187;
                    double y_15199 = x_15183 * x_15188;
                    double x_15200 = x_15198 + y_15199;
                    double res_15201 = value_15190 * x_15200;
                    double x_15202 = x_15182 * x_15187;
                    double y_15203 = x_15184 * x_15188;
                    double x_15204 = x_15202 + y_15203;
                    double res_15205 = value_15190 * x_15204;
                    
                    x_15181 = res_15193;
                    x_15182 = res_15197;
                    x_15183 = res_15201;
                    x_15184 = res_15205;
                }
            }
            // write final result
            {
                ((__local double *) mem_16417)[local_tid_16608] = x_15181;
                ((__local double *) mem_16420)[local_tid_16608] = x_15182;
                ((__local double *) mem_16423)[local_tid_16608] = x_15183;
                ((__local double *) mem_16426)[local_tid_16608] = x_15184;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16608, 32) == 0) {
            ((__local double *) mem_16417)[local_tid_16608] = x_15185;
            ((__local double *) mem_16420)[local_tid_16608] = x_15186;
            ((__local double *) mem_16423)[local_tid_16608] = x_15187;
            ((__local double *) mem_16426)[local_tid_16608] = x_15188;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16430;
    
    mem_16430 = (__local char *) mem_16430_backing_4;
    
    int32_t gtid_14661 = ltid_pre_16612;
    int32_t phys_tid_14662 = local_tid_16608;
    
    if (slt32(gtid_14661, m_13844)) {
        double x_15246 = ((__local double *) mem_16417)[gtid_14661];
        double x_15247 = ((__local double *) mem_16420)[gtid_14661];
        double x_15248 = ((__local double *) mem_16423)[gtid_14661];
        double x_15249 = ((__local double *) mem_16426)[gtid_14661];
        double value_15251 = 1.0 / x_15246;
        double res_15254 = x_15246 * value_15251;
        double res_15258 = x_15247 * value_15251;
        double res_15262 = x_15248 * value_15251;
        double res_15266 = x_15249 * value_15251;
        double x_15267 = as_transformed_row_15167 * res_15254;
        double x_15268 = res_15258 + x_15267;
        double x_15269 = as_transformed_row_15167 * res_15262;
        double y_15270 = res_15266 + x_15269;
        double res_15271 = x_15268 / y_15270;
        
        ((__local double *) mem_16430)[gtid_14661] = res_15271;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16434;
    
    mem_16434 = (__local char *) mem_16434_backing_5;
    
    __local char *mem_16437;
    
    mem_16437 = (__local char *) mem_16437_backing_6;
    
    int32_t gtid_14689 = ltid_pre_16612;
    int32_t phys_tid_14690 = local_tid_16608;
    
    if (slt32(gtid_14689, m_13844)) {
        bool cond_15295 = slt32(0, gtid_14689);
        double res_15296;
        
        if (cond_15295) {
            double x_elem_15293 = ((__global double *) y_mem_16412)[gtid_14656 *
                                                                    m_13850 +
                                                                    gtid_14689];
            
            res_15296 = x_elem_15293;
        } else {
            res_15296 = 0.0;
        }
        
        double res_15297;
        
        if (cond_15295) {
            double x_elem_15294 = ((__global double *) a_mem_16409)[gtid_14656 *
                                                                    m_13844 +
                                                                    gtid_14689];
            int32_t i_15298 = sub32(gtid_14689, 1);
            double y_15299 = ((__local double *) mem_16430)[i_15298];
            double y_15300 = x_elem_15294 / y_15299;
            double res_15301 = 0.0 - y_15300;
            
            res_15297 = res_15301;
        } else {
            res_15297 = 1.0;
        }
        ((__local double *) mem_16434)[gtid_14689] = res_15296;
        ((__local double *) mem_16437)[gtid_14689] = res_15297;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_16648;
    
    dims_flat_16648 = m_13844;
    
    double x_15285;
    double x_15286;
    double x_15287;
    double x_15288;
    double x_16651;
    double x_16652;
    double x_16653;
    double x_16654;
    int32_t skip_threads_16658;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16608, m_13844)) {
            x_15287 = ((volatile __local double *) mem_16434)[local_tid_16608];
            x_15288 = ((volatile __local double *) mem_16437)[local_tid_16608];
            if ((local_tid_16608 - squot32(local_tid_16608, 32) * 32) == 0) {
                x_15285 = x_15287;
                x_15286 = x_15288;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16658 = 1;
        while (slt32(skip_threads_16658, 32)) {
            if (sle32(skip_threads_16658, local_tid_16608 -
                      squot32(local_tid_16608, 32) * 32) &&
                slt32(local_tid_16608, m_13844)) {
                // read operands
                {
                    x_15285 = ((volatile __local
                                double *) mem_16434)[local_tid_16608 -
                                                     skip_threads_16658];
                    x_15286 = ((volatile __local
                                double *) mem_16437)[local_tid_16608 -
                                                     skip_threads_16658];
                }
                // perform operation
                {
                    bool inactive_16659 = slt32(srem32(local_tid_16608,
                                                       m_13844),
                                                local_tid_16608 -
                                                (local_tid_16608 -
                                                 skip_threads_16658));
                    
                    if (inactive_16659) {
                        x_15285 = x_15287;
                        x_15286 = x_15288;
                    }
                    if (!inactive_16659) {
                        double y_15289 = x_15285 * x_15288;
                        double res_15290 = x_15287 + y_15289;
                        double res_15291 = x_15286 * x_15288;
                        
                        x_15285 = res_15290;
                        x_15286 = res_15291;
                    }
                }
            }
            if (sle32(wave_sizze_16610, skip_threads_16658)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16658, local_tid_16608 -
                      squot32(local_tid_16608, 32) * 32) &&
                slt32(local_tid_16608, m_13844)) {
                // write result
                {
                    ((volatile __local double *) mem_16434)[local_tid_16608] =
                        x_15285;
                    x_15287 = x_15285;
                    ((volatile __local double *) mem_16437)[local_tid_16608] =
                        x_15286;
                    x_15288 = x_15286;
                }
            }
            if (sle32(wave_sizze_16610, skip_threads_16658)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16658 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16608 - squot32(local_tid_16608, 32) * 32) == 31 &&
            slt32(local_tid_16608, m_13844)) {
            ((volatile __local double *) mem_16434)[squot32(local_tid_16608,
                                                            32)] = x_15285;
            ((volatile __local double *) mem_16437)[squot32(local_tid_16608,
                                                            32)] = x_15286;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16660;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16608, 32) == 0 && slt32(local_tid_16608,
                                                           m_13844)) {
                x_16653 = ((volatile __local
                            double *) mem_16434)[local_tid_16608];
                x_16654 = ((volatile __local
                            double *) mem_16437)[local_tid_16608];
                if ((local_tid_16608 - squot32(local_tid_16608, 32) * 32) ==
                    0) {
                    x_16651 = x_16653;
                    x_16652 = x_16654;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16660 = 1;
            while (slt32(skip_threads_16660, 32)) {
                if (sle32(skip_threads_16660, local_tid_16608 -
                          squot32(local_tid_16608, 32) * 32) &&
                    (squot32(local_tid_16608, 32) == 0 && slt32(local_tid_16608,
                                                                m_13844))) {
                    // read operands
                    {
                        x_16651 = ((volatile __local
                                    double *) mem_16434)[local_tid_16608 -
                                                         skip_threads_16660];
                        x_16652 = ((volatile __local
                                    double *) mem_16437)[local_tid_16608 -
                                                         skip_threads_16660];
                    }
                    // perform operation
                    {
                        bool inactive_16661 = slt32(srem32(local_tid_16608 *
                                                           32 + 32 - 1,
                                                           m_13844),
                                                    local_tid_16608 * 32 + 32 -
                                                    1 - ((local_tid_16608 -
                                                          skip_threads_16660) *
                                                         32 + 32 - 1));
                        
                        if (inactive_16661) {
                            x_16651 = x_16653;
                            x_16652 = x_16654;
                        }
                        if (!inactive_16661) {
                            double y_16655 = x_16651 * x_16654;
                            double res_16656 = x_16653 + y_16655;
                            double res_16657 = x_16652 * x_16654;
                            
                            x_16651 = res_16656;
                            x_16652 = res_16657;
                        }
                    }
                }
                if (sle32(wave_sizze_16610, skip_threads_16660)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16660, local_tid_16608 -
                          squot32(local_tid_16608, 32) * 32) &&
                    (squot32(local_tid_16608, 32) == 0 && slt32(local_tid_16608,
                                                                m_13844))) {
                    // write result
                    {
                        ((volatile __local
                          double *) mem_16434)[local_tid_16608] = x_16651;
                        x_16653 = x_16651;
                        ((volatile __local
                          double *) mem_16437)[local_tid_16608] = x_16652;
                        x_16654 = x_16652;
                    }
                }
                if (sle32(wave_sizze_16610, skip_threads_16660)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16660 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16608, 32) == 0 || !slt32(local_tid_16608,
                                                          m_13844))) {
            // read operands
            {
                x_15287 = x_15285;
                x_15288 = x_15286;
                x_15285 = ((__local
                            double *) mem_16434)[squot32(local_tid_16608, 32) -
                                                 1];
                x_15286 = ((__local
                            double *) mem_16437)[squot32(local_tid_16608, 32) -
                                                 1];
            }
            // perform operation
            {
                bool inactive_16662 = slt32(srem32(local_tid_16608, m_13844),
                                            local_tid_16608 -
                                            (squot32(local_tid_16608, 32) * 32 -
                                             1));
                
                if (inactive_16662) {
                    x_15285 = x_15287;
                    x_15286 = x_15288;
                }
                if (!inactive_16662) {
                    double y_15289 = x_15285 * x_15288;
                    double res_15290 = x_15287 + y_15289;
                    double res_15291 = x_15286 * x_15288;
                    
                    x_15285 = res_15290;
                    x_15286 = res_15291;
                }
            }
            // write final result
            {
                ((__local double *) mem_16434)[local_tid_16608] = x_15285;
                ((__local double *) mem_16437)[local_tid_16608] = x_15286;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16608, 32) == 0) {
            ((__local double *) mem_16434)[local_tid_16608] = x_15287;
            ((__local double *) mem_16437)[local_tid_16608] = x_15288;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16441;
    
    mem_16441 = (__local char *) mem_16441_backing_7;
    
    int32_t gtid_14691 = ltid_pre_16612;
    int32_t phys_tid_14692 = local_tid_16608;
    
    if (slt32(gtid_14691, m_13844)) {
        double x_15312 = ((__local double *) mem_16434)[gtid_14691];
        double x_15313 = ((__local double *) mem_16437)[gtid_14691];
        double y_15317 = as_transformed_row_15168 * x_15313;
        double res_15318 = x_15312 + y_15317;
        
        ((__local double *) mem_16441)[gtid_14691] = res_15318;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    double x_15322 = ((__local double *) mem_16441)[i_13871];
    double y_15323 = ((__local double *) mem_16430)[i_13871];
    double yn_15324 = x_15322 / y_15323;
    __local char *mem_16445;
    
    mem_16445 = (__local char *) mem_16445_backing_8;
    
    __local char *mem_16448;
    
    mem_16448 = (__local char *) mem_16448_backing_9;
    
    int32_t gtid_14700 = ltid_pre_16612;
    int32_t phys_tid_14701 = local_tid_16608;
    
    if (slt32(gtid_14700, m_13844)) {
        int32_t x_15339 = sub32(m_13844, gtid_14700);
        int32_t i_15340 = sub32(x_15339, 1);
        bool cond_15341 = slt32(0, gtid_14700);
        double res_15342;
        double res_15343;
        
        if (cond_15341) {
            double x_15344 = ((__local double *) mem_16441)[i_15340];
            double y_15345 = ((__local double *) mem_16430)[i_15340];
            double res_15346 = x_15344 / y_15345;
            double x_15347 = ((__global double *) c_mem_16411)[gtid_14656 *
                                                               m_13848 +
                                                               i_15340];
            double y_15348 = x_15347 / y_15345;
            double res_15349 = 0.0 - y_15348;
            
            res_15342 = res_15346;
            res_15343 = res_15349;
        } else {
            res_15342 = 0.0;
            res_15343 = 1.0;
        }
        ((__local double *) mem_16445)[gtid_14700] = res_15342;
        ((__local double *) mem_16448)[gtid_14700] = res_15343;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_16663;
    
    dims_flat_16663 = m_13844;
    
    double x_15331;
    double x_15332;
    double x_15333;
    double x_15334;
    double x_16666;
    double x_16667;
    double x_16668;
    double x_16669;
    int32_t skip_threads_16673;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16608, m_13844)) {
            x_15333 = ((volatile __local double *) mem_16445)[local_tid_16608];
            x_15334 = ((volatile __local double *) mem_16448)[local_tid_16608];
            if ((local_tid_16608 - squot32(local_tid_16608, 32) * 32) == 0) {
                x_15331 = x_15333;
                x_15332 = x_15334;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16673 = 1;
        while (slt32(skip_threads_16673, 32)) {
            if (sle32(skip_threads_16673, local_tid_16608 -
                      squot32(local_tid_16608, 32) * 32) &&
                slt32(local_tid_16608, m_13844)) {
                // read operands
                {
                    x_15331 = ((volatile __local
                                double *) mem_16445)[local_tid_16608 -
                                                     skip_threads_16673];
                    x_15332 = ((volatile __local
                                double *) mem_16448)[local_tid_16608 -
                                                     skip_threads_16673];
                }
                // perform operation
                {
                    bool inactive_16674 = slt32(srem32(local_tid_16608,
                                                       m_13844),
                                                local_tid_16608 -
                                                (local_tid_16608 -
                                                 skip_threads_16673));
                    
                    if (inactive_16674) {
                        x_15331 = x_15333;
                        x_15332 = x_15334;
                    }
                    if (!inactive_16674) {
                        double y_15335 = x_15331 * x_15334;
                        double res_15336 = x_15333 + y_15335;
                        double res_15337 = x_15332 * x_15334;
                        
                        x_15331 = res_15336;
                        x_15332 = res_15337;
                    }
                }
            }
            if (sle32(wave_sizze_16610, skip_threads_16673)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16673, local_tid_16608 -
                      squot32(local_tid_16608, 32) * 32) &&
                slt32(local_tid_16608, m_13844)) {
                // write result
                {
                    ((volatile __local double *) mem_16445)[local_tid_16608] =
                        x_15331;
                    x_15333 = x_15331;
                    ((volatile __local double *) mem_16448)[local_tid_16608] =
                        x_15332;
                    x_15334 = x_15332;
                }
            }
            if (sle32(wave_sizze_16610, skip_threads_16673)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16673 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16608 - squot32(local_tid_16608, 32) * 32) == 31 &&
            slt32(local_tid_16608, m_13844)) {
            ((volatile __local double *) mem_16445)[squot32(local_tid_16608,
                                                            32)] = x_15331;
            ((volatile __local double *) mem_16448)[squot32(local_tid_16608,
                                                            32)] = x_15332;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16675;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16608, 32) == 0 && slt32(local_tid_16608,
                                                           m_13844)) {
                x_16668 = ((volatile __local
                            double *) mem_16445)[local_tid_16608];
                x_16669 = ((volatile __local
                            double *) mem_16448)[local_tid_16608];
                if ((local_tid_16608 - squot32(local_tid_16608, 32) * 32) ==
                    0) {
                    x_16666 = x_16668;
                    x_16667 = x_16669;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16675 = 1;
            while (slt32(skip_threads_16675, 32)) {
                if (sle32(skip_threads_16675, local_tid_16608 -
                          squot32(local_tid_16608, 32) * 32) &&
                    (squot32(local_tid_16608, 32) == 0 && slt32(local_tid_16608,
                                                                m_13844))) {
                    // read operands
                    {
                        x_16666 = ((volatile __local
                                    double *) mem_16445)[local_tid_16608 -
                                                         skip_threads_16675];
                        x_16667 = ((volatile __local
                                    double *) mem_16448)[local_tid_16608 -
                                                         skip_threads_16675];
                    }
                    // perform operation
                    {
                        bool inactive_16676 = slt32(srem32(local_tid_16608 *
                                                           32 + 32 - 1,
                                                           m_13844),
                                                    local_tid_16608 * 32 + 32 -
                                                    1 - ((local_tid_16608 -
                                                          skip_threads_16675) *
                                                         32 + 32 - 1));
                        
                        if (inactive_16676) {
                            x_16666 = x_16668;
                            x_16667 = x_16669;
                        }
                        if (!inactive_16676) {
                            double y_16670 = x_16666 * x_16669;
                            double res_16671 = x_16668 + y_16670;
                            double res_16672 = x_16667 * x_16669;
                            
                            x_16666 = res_16671;
                            x_16667 = res_16672;
                        }
                    }
                }
                if (sle32(wave_sizze_16610, skip_threads_16675)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16675, local_tid_16608 -
                          squot32(local_tid_16608, 32) * 32) &&
                    (squot32(local_tid_16608, 32) == 0 && slt32(local_tid_16608,
                                                                m_13844))) {
                    // write result
                    {
                        ((volatile __local
                          double *) mem_16445)[local_tid_16608] = x_16666;
                        x_16668 = x_16666;
                        ((volatile __local
                          double *) mem_16448)[local_tid_16608] = x_16667;
                        x_16669 = x_16667;
                    }
                }
                if (sle32(wave_sizze_16610, skip_threads_16675)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16675 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16608, 32) == 0 || !slt32(local_tid_16608,
                                                          m_13844))) {
            // read operands
            {
                x_15333 = x_15331;
                x_15334 = x_15332;
                x_15331 = ((__local
                            double *) mem_16445)[squot32(local_tid_16608, 32) -
                                                 1];
                x_15332 = ((__local
                            double *) mem_16448)[squot32(local_tid_16608, 32) -
                                                 1];
            }
            // perform operation
            {
                bool inactive_16677 = slt32(srem32(local_tid_16608, m_13844),
                                            local_tid_16608 -
                                            (squot32(local_tid_16608, 32) * 32 -
                                             1));
                
                if (inactive_16677) {
                    x_15331 = x_15333;
                    x_15332 = x_15334;
                }
                if (!inactive_16677) {
                    double y_15335 = x_15331 * x_15334;
                    double res_15336 = x_15333 + y_15335;
                    double res_15337 = x_15332 * x_15334;
                    
                    x_15331 = res_15336;
                    x_15332 = res_15337;
                }
            }
            // write final result
            {
                ((__local double *) mem_16445)[local_tid_16608] = x_15331;
                ((__local double *) mem_16448)[local_tid_16608] = x_15332;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16608, 32) == 0) {
            ((__local double *) mem_16445)[local_tid_16608] = x_15333;
            ((__local double *) mem_16448)[local_tid_16608] = x_15334;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16452;
    
    mem_16452 = (__local char *) mem_16452_backing_10;
    
    int32_t gtid_14702 = ltid_pre_16612;
    int32_t phys_tid_14703 = local_tid_16608;
    
    if (slt32(gtid_14702, m_13844)) {
        double x_15360 = ((__local double *) mem_16445)[gtid_14702];
        double x_15361 = ((__local double *) mem_16448)[gtid_14702];
        double y_15365 = yn_15324 * x_15361;
        double res_15366 = x_15360 + y_15365;
        
        ((__local double *) mem_16452)[gtid_14702] = res_15366;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16456;
    
    mem_16456 = (__local char *) mem_16456_backing_11;
    
    int32_t gtid_14711 = ltid_pre_16612;
    int32_t phys_tid_14712 = local_tid_16608;
    
    if (slt32(gtid_14711, m_13844)) {
        int32_t x_15372 = sub32(m_13844, gtid_14711);
        int32_t i_15373 = sub32(x_15372, 1);
        double res_15374 = ((__local double *) mem_16452)[i_15373];
        
        ((__local double *) mem_16456)[gtid_14711] = res_15374;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ((__global double *) mem_16461)[gtid_14656 * m_13844 + local_tid_16608] =
        ((__local double *) mem_16456)[local_tid_16608];
    barrier(CLK_LOCAL_MEM_FENCE);
    
  error_7:
    return;
}
__kernel void tridagParFlatziscan_stage1_14146(__global int *global_failure,
                                               __local volatile
                                               int64_t *scan_arr_mem_16636_backing_aligned_0,
                                               __local volatile
                                               int64_t *scan_arr_mem_16634_backing_aligned_1,
                                               int32_t n_13426, __global
                                               unsigned char *mem_16418,
                                               __global
                                               unsigned char *mem_16423,
                                               __global
                                               unsigned char *mem_16426,
                                               int32_t num_threads_16628)
{
    #define segscan_group_sizze_14141 (tridagParFlatzisegscan_group_sizze_14140)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16636_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16636_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16634_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16634_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16629;
    int32_t local_tid_16630;
    int32_t group_sizze_16633;
    int32_t wave_sizze_16632;
    int32_t group_tid_16631;
    
    global_tid_16629 = get_global_id(0);
    local_tid_16630 = get_local_id(0);
    group_sizze_16633 = get_local_size(0);
    wave_sizze_16632 = LOCKSTEP_WIDTH;
    group_tid_16631 = get_group_id(0);
    
    int32_t phys_tid_14146;
    
    phys_tid_14146 = global_tid_16629;
    
    __local char *scan_arr_mem_16634;
    __local char *scan_arr_mem_16636;
    
    scan_arr_mem_16634 = (__local char *) scan_arr_mem_16634_backing_0;
    scan_arr_mem_16636 = (__local char *) scan_arr_mem_16636_backing_1;
    
    int32_t x_13478;
    double x_13479;
    int32_t x_13480;
    double x_13481;
    
    x_13478 = 0;
    x_13479 = 0.0;
    for (int32_t j_16638 = 0; j_16638 < sdiv_up32(n_13426, num_threads_16628);
         j_16638++) {
        int32_t chunk_offset_16639 = segscan_group_sizze_14141 * j_16638 +
                group_tid_16631 * (segscan_group_sizze_14141 *
                                   sdiv_up32(n_13426, num_threads_16628));
        int32_t flat_idx_16640 = chunk_offset_16639 + local_tid_16630;
        int32_t gtid_14145 = flat_idx_16640;
        
        // threads in bounds read input
        {
            if (slt32(gtid_14145, n_13426)) {
                double x_13486 = ((__global double *) mem_16418)[gtid_14145];
                bool cond_13487 = 0.0 < x_13486;
                int32_t res_13488 = btoi_bool_i32(cond_13487);
                
                // write to-scan values to parameters
                {
                    x_13480 = res_13488;
                    x_13481 = x_13486;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt32(gtid_14145, n_13426)) {
                    x_13480 = 0;
                    x_13481 = 0.0;
                }
            }
            // combine with carry and write to local memory
            {
                int32_t f_13482 = x_13478 | x_13480;
                bool cond_13483 = slt32(0, x_13480);
                double res_13484;
                
                if (cond_13483) {
                    res_13484 = x_13481;
                } else {
                    double res_13485 = x_13479 + x_13481;
                    
                    res_13484 = res_13485;
                }
                ((__local int32_t *) scan_arr_mem_16634)[local_tid_16630] =
                    f_13482;
                ((__local double *) scan_arr_mem_16636)[local_tid_16630] =
                    res_13484;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t x_16641;
            double x_16642;
            int32_t x_16643;
            double x_16644;
            int32_t x_16649;
            double x_16650;
            int32_t x_16651;
            double x_16652;
            int32_t skip_threads_16657;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_16630, segscan_group_sizze_14141)) {
                    x_16643 = ((volatile __local
                                int32_t *) scan_arr_mem_16634)[local_tid_16630];
                    x_16644 = ((volatile __local
                                double *) scan_arr_mem_16636)[local_tid_16630];
                    if ((local_tid_16630 - squot32(local_tid_16630, 32) * 32) ==
                        0) {
                        x_16641 = x_16643;
                        x_16642 = x_16644;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_16657 = 1;
                while (slt32(skip_threads_16657, 32)) {
                    if (sle32(skip_threads_16657, local_tid_16630 -
                              squot32(local_tid_16630, 32) * 32) &&
                        slt32(local_tid_16630, segscan_group_sizze_14141)) {
                        // read operands
                        {
                            x_16641 = ((volatile __local
                                        int32_t *) scan_arr_mem_16634)[local_tid_16630 -
                                                                       skip_threads_16657];
                            x_16642 = ((volatile __local
                                        double *) scan_arr_mem_16636)[local_tid_16630 -
                                                                      skip_threads_16657];
                        }
                        // perform operation
                        {
                            int32_t f_16645 = x_16641 | x_16643;
                            bool cond_16646 = slt32(0, x_16643);
                            double res_16647;
                            
                            if (cond_16646) {
                                res_16647 = x_16644;
                            } else {
                                double res_16648 = x_16642 + x_16644;
                                
                                res_16647 = res_16648;
                            }
                            x_16641 = f_16645;
                            x_16642 = res_16647;
                        }
                    }
                    if (sle32(wave_sizze_16632, skip_threads_16657)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_16657, local_tid_16630 -
                              squot32(local_tid_16630, 32) * 32) &&
                        slt32(local_tid_16630, segscan_group_sizze_14141)) {
                        // write result
                        {
                            ((volatile __local
                              int32_t *) scan_arr_mem_16634)[local_tid_16630] =
                                x_16641;
                            x_16643 = x_16641;
                            ((volatile __local
                              double *) scan_arr_mem_16636)[local_tid_16630] =
                                x_16642;
                            x_16644 = x_16642;
                        }
                    }
                    if (sle32(wave_sizze_16632, skip_threads_16657)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_16657 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_16630 - squot32(local_tid_16630, 32) * 32) ==
                    31 && slt32(local_tid_16630, segscan_group_sizze_14141)) {
                    ((volatile __local
                      int32_t *) scan_arr_mem_16634)[squot32(local_tid_16630,
                                                             32)] = x_16641;
                    ((volatile __local
                      double *) scan_arr_mem_16636)[squot32(local_tid_16630,
                                                            32)] = x_16642;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_16658;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_16630, 32) == 0 &&
                        slt32(local_tid_16630, segscan_group_sizze_14141)) {
                        x_16651 = ((volatile __local
                                    int32_t *) scan_arr_mem_16634)[local_tid_16630];
                        x_16652 = ((volatile __local
                                    double *) scan_arr_mem_16636)[local_tid_16630];
                        if ((local_tid_16630 - squot32(local_tid_16630, 32) *
                             32) == 0) {
                            x_16649 = x_16651;
                            x_16650 = x_16652;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_16658 = 1;
                    while (slt32(skip_threads_16658, 32)) {
                        if (sle32(skip_threads_16658, local_tid_16630 -
                                  squot32(local_tid_16630, 32) * 32) &&
                            (squot32(local_tid_16630, 32) == 0 &&
                             slt32(local_tid_16630,
                                   segscan_group_sizze_14141))) {
                            // read operands
                            {
                                x_16649 = ((volatile __local
                                            int32_t *) scan_arr_mem_16634)[local_tid_16630 -
                                                                           skip_threads_16658];
                                x_16650 = ((volatile __local
                                            double *) scan_arr_mem_16636)[local_tid_16630 -
                                                                          skip_threads_16658];
                            }
                            // perform operation
                            {
                                int32_t f_16653 = x_16649 | x_16651;
                                bool cond_16654 = slt32(0, x_16651);
                                double res_16655;
                                
                                if (cond_16654) {
                                    res_16655 = x_16652;
                                } else {
                                    double res_16656 = x_16650 + x_16652;
                                    
                                    res_16655 = res_16656;
                                }
                                x_16649 = f_16653;
                                x_16650 = res_16655;
                            }
                        }
                        if (sle32(wave_sizze_16632, skip_threads_16658)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_16658, local_tid_16630 -
                                  squot32(local_tid_16630, 32) * 32) &&
                            (squot32(local_tid_16630, 32) == 0 &&
                             slt32(local_tid_16630,
                                   segscan_group_sizze_14141))) {
                            // write result
                            {
                                ((volatile __local
                                  int32_t *) scan_arr_mem_16634)[local_tid_16630] =
                                    x_16649;
                                x_16651 = x_16649;
                                ((volatile __local
                                  double *) scan_arr_mem_16636)[local_tid_16630] =
                                    x_16650;
                                x_16652 = x_16650;
                            }
                        }
                        if (sle32(wave_sizze_16632, skip_threads_16658)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_16658 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_16630, 32) == 0 ||
                      !slt32(local_tid_16630, segscan_group_sizze_14141))) {
                    // read operands
                    {
                        x_16643 = x_16641;
                        x_16644 = x_16642;
                        x_16641 = ((__local
                                    int32_t *) scan_arr_mem_16634)[squot32(local_tid_16630,
                                                                           32) -
                                                                   1];
                        x_16642 = ((__local
                                    double *) scan_arr_mem_16636)[squot32(local_tid_16630,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        int32_t f_16645 = x_16641 | x_16643;
                        bool cond_16646 = slt32(0, x_16643);
                        double res_16647;
                        
                        if (cond_16646) {
                            res_16647 = x_16644;
                        } else {
                            double res_16648 = x_16642 + x_16644;
                            
                            res_16647 = res_16648;
                        }
                        x_16641 = f_16645;
                        x_16642 = res_16647;
                    }
                    // write final result
                    {
                        ((__local
                          int32_t *) scan_arr_mem_16634)[local_tid_16630] =
                            x_16641;
                        ((__local
                          double *) scan_arr_mem_16636)[local_tid_16630] =
                            x_16642;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_16630, 32) == 0) {
                    ((__local int32_t *) scan_arr_mem_16634)[local_tid_16630] =
                        x_16643;
                    ((__local double *) scan_arr_mem_16636)[local_tid_16630] =
                        x_16644;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_14145, n_13426)) {
                    ((__global int32_t *) mem_16423)[gtid_14145] = ((__local
                                                                     int32_t *) scan_arr_mem_16634)[local_tid_16630];
                    ((__global double *) mem_16426)[gtid_14145] = ((__local
                                                                    double *) scan_arr_mem_16636)[local_tid_16630];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_16659 = 0;
                bool should_load_carry_16660 = local_tid_16630 == 0 &&
                     !crosses_segment_16659;
                
                if (should_load_carry_16660) {
                    x_13478 = ((__local
                                int32_t *) scan_arr_mem_16634)[segscan_group_sizze_14141 -
                                                               1];
                    x_13479 = ((__local
                                double *) scan_arr_mem_16636)[segscan_group_sizze_14141 -
                                                              1];
                }
                if (!should_load_carry_16660) {
                    x_13478 = 0;
                    x_13479 = 0.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_14141
}
__kernel void tridagParFlatziscan_stage1_14155(__global int *global_failure,
                                               __local volatile
                                               int64_t *scan_arr_mem_16710_backing_aligned_0,
                                               __local volatile
                                               int64_t *scan_arr_mem_16708_backing_aligned_1,
                                               __local volatile
                                               int64_t *scan_arr_mem_16706_backing_aligned_2,
                                               __local volatile
                                               int64_t *scan_arr_mem_16704_backing_aligned_3,
                                               __local volatile
                                               int64_t *scan_arr_mem_16702_backing_aligned_4,
                                               int32_t n_13426,
                                               int32_t segSizze_13434, __global
                                               unsigned char *a_mem_16409,
                                               __global
                                               unsigned char *b_mem_16410,
                                               __global
                                               unsigned char *c_mem_16411,
                                               __global
                                               unsigned char *mem_16415,
                                               __global
                                               unsigned char *mem_16430,
                                               __global
                                               unsigned char *mem_16433,
                                               __global
                                               unsigned char *mem_16436,
                                               __global
                                               unsigned char *mem_16439,
                                               __global
                                               unsigned char *mem_16442,
                                               int32_t num_threads_16696)
{
    #define segscan_group_sizze_14150 (tridagParFlatzisegscan_group_sizze_14149)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16710_backing_4 =
                          (__local volatile
                           char *) scan_arr_mem_16710_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16708_backing_3 =
                          (__local volatile
                           char *) scan_arr_mem_16708_backing_aligned_1;
    __local volatile char *restrict scan_arr_mem_16706_backing_2 =
                          (__local volatile
                           char *) scan_arr_mem_16706_backing_aligned_2;
    __local volatile char *restrict scan_arr_mem_16704_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16704_backing_aligned_3;
    __local volatile char *restrict scan_arr_mem_16702_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16702_backing_aligned_4;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16697;
    int32_t local_tid_16698;
    int32_t group_sizze_16701;
    int32_t wave_sizze_16700;
    int32_t group_tid_16699;
    
    global_tid_16697 = get_global_id(0);
    local_tid_16698 = get_local_id(0);
    group_sizze_16701 = get_local_size(0);
    wave_sizze_16700 = LOCKSTEP_WIDTH;
    group_tid_16699 = get_group_id(0);
    
    int32_t phys_tid_14155;
    
    phys_tid_14155 = global_tid_16697;
    
    __local char *scan_arr_mem_16702;
    __local char *scan_arr_mem_16704;
    __local char *scan_arr_mem_16706;
    __local char *scan_arr_mem_16708;
    __local char *scan_arr_mem_16710;
    
    scan_arr_mem_16702 = (__local char *) scan_arr_mem_16702_backing_0;
    scan_arr_mem_16704 = (__local char *) scan_arr_mem_16704_backing_1;
    scan_arr_mem_16706 = (__local char *) scan_arr_mem_16706_backing_2;
    scan_arr_mem_16708 = (__local char *) scan_arr_mem_16708_backing_3;
    scan_arr_mem_16710 = (__local char *) scan_arr_mem_16710_backing_4;
    
    int32_t x_13504;
    double x_13505;
    double x_13506;
    double x_13507;
    double x_13508;
    int32_t x_13509;
    double x_13510;
    double x_13511;
    double x_13512;
    double x_13513;
    
    x_13504 = 0;
    x_13505 = 1.0;
    x_13506 = 0.0;
    x_13507 = 0.0;
    x_13508 = 1.0;
    for (int32_t j_16712 = 0; j_16712 < sdiv_up32(n_13426, num_threads_16696);
         j_16712++) {
        int32_t chunk_offset_16713 = segscan_group_sizze_14150 * j_16712 +
                group_tid_16699 * (segscan_group_sizze_14150 *
                                   sdiv_up32(n_13426, num_threads_16696));
        int32_t flat_idx_16714 = chunk_offset_16713 + local_tid_16698;
        int32_t gtid_14154 = flat_idx_16714;
        
        // threads in bounds read input
        {
            if (slt32(gtid_14154, n_13426)) {
                int32_t x_13540 = ((__global int32_t *) mem_16415)[gtid_14154];
                int32_t y_13541 = smod32(gtid_14154, segSizze_13434);
                bool cond_13542 = slt32(0, y_13541);
                double res_13543;
                
                if (cond_13542) {
                    res_13543 = 1.0;
                } else {
                    res_13543 = 0.0;
                }
                
                double res_13544;
                
                if (cond_13542) {
                    res_13544 = 0.0;
                } else {
                    res_13544 = 1.0;
                }
                
                double res_13545;
                
                if (cond_13542) {
                    double b_elem_13538 = ((__global
                                            double *) b_mem_16410)[gtid_14154];
                    
                    res_13545 = b_elem_13538;
                } else {
                    res_13545 = 1.0;
                }
                
                double res_13546;
                
                if (cond_13542) {
                    double a_elem_13539 = ((__global
                                            double *) a_mem_16409)[gtid_14154];
                    int32_t i_13547 = sub32(gtid_14154, 1);
                    double y_13548 = ((__global double *) c_mem_16411)[i_13547];
                    double y_13549 = a_elem_13539 * y_13548;
                    double res_13550 = 0.0 - y_13549;
                    
                    res_13546 = res_13550;
                } else {
                    res_13546 = 0.0;
                }
                // write to-scan values to parameters
                {
                    x_13509 = x_13540;
                    x_13510 = res_13545;
                    x_13511 = res_13546;
                    x_13512 = res_13543;
                    x_13513 = res_13544;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt32(gtid_14154, n_13426)) {
                    x_13509 = 0;
                    x_13510 = 1.0;
                    x_13511 = 0.0;
                    x_13512 = 0.0;
                    x_13513 = 1.0;
                }
            }
            // combine with carry and write to local memory
            {
                int32_t f_13514 = x_13504 | x_13509;
                bool cond_13515 = slt32(0, x_13509);
                double res_13516;
                double res_13517;
                double res_13518;
                double res_13519;
                
                if (cond_13515) {
                    res_13516 = x_13510;
                    res_13517 = x_13511;
                    res_13518 = x_13512;
                    res_13519 = x_13513;
                } else {
                    double y_13520 = x_13505 * x_13510;
                    double value_13521 = 1.0 / y_13520;
                    double y_13522 = x_13507 * x_13511;
                    double x_13523 = y_13520 + y_13522;
                    double res_13524 = value_13521 * x_13523;
                    double x_13525 = x_13506 * x_13510;
                    double y_13526 = x_13508 * x_13511;
                    double x_13527 = x_13525 + y_13526;
                    double res_13528 = value_13521 * x_13527;
                    double x_13529 = x_13505 * x_13512;
                    double y_13530 = x_13507 * x_13513;
                    double x_13531 = x_13529 + y_13530;
                    double res_13532 = value_13521 * x_13531;
                    double x_13533 = x_13506 * x_13512;
                    double y_13534 = x_13508 * x_13513;
                    double x_13535 = x_13533 + y_13534;
                    double res_13536 = value_13521 * x_13535;
                    
                    res_13516 = res_13524;
                    res_13517 = res_13528;
                    res_13518 = res_13532;
                    res_13519 = res_13536;
                }
                ((__local int32_t *) scan_arr_mem_16702)[local_tid_16698] =
                    f_13514;
                ((__local double *) scan_arr_mem_16704)[local_tid_16698] =
                    res_13516;
                ((__local double *) scan_arr_mem_16706)[local_tid_16698] =
                    res_13517;
                ((__local double *) scan_arr_mem_16708)[local_tid_16698] =
                    res_13518;
                ((__local double *) scan_arr_mem_16710)[local_tid_16698] =
                    res_13519;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t x_16715;
            double x_16716;
            double x_16717;
            double x_16718;
            double x_16719;
            int32_t x_16720;
            double x_16721;
            double x_16722;
            double x_16723;
            double x_16724;
            int32_t x_16748;
            double x_16749;
            double x_16750;
            double x_16751;
            double x_16752;
            int32_t x_16753;
            double x_16754;
            double x_16755;
            double x_16756;
            double x_16757;
            int32_t skip_threads_16781;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_16698, segscan_group_sizze_14150)) {
                    x_16720 = ((volatile __local
                                int32_t *) scan_arr_mem_16702)[local_tid_16698];
                    x_16721 = ((volatile __local
                                double *) scan_arr_mem_16704)[local_tid_16698];
                    x_16722 = ((volatile __local
                                double *) scan_arr_mem_16706)[local_tid_16698];
                    x_16723 = ((volatile __local
                                double *) scan_arr_mem_16708)[local_tid_16698];
                    x_16724 = ((volatile __local
                                double *) scan_arr_mem_16710)[local_tid_16698];
                    if ((local_tid_16698 - squot32(local_tid_16698, 32) * 32) ==
                        0) {
                        x_16715 = x_16720;
                        x_16716 = x_16721;
                        x_16717 = x_16722;
                        x_16718 = x_16723;
                        x_16719 = x_16724;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_16781 = 1;
                while (slt32(skip_threads_16781, 32)) {
                    if (sle32(skip_threads_16781, local_tid_16698 -
                              squot32(local_tid_16698, 32) * 32) &&
                        slt32(local_tid_16698, segscan_group_sizze_14150)) {
                        // read operands
                        {
                            x_16715 = ((volatile __local
                                        int32_t *) scan_arr_mem_16702)[local_tid_16698 -
                                                                       skip_threads_16781];
                            x_16716 = ((volatile __local
                                        double *) scan_arr_mem_16704)[local_tid_16698 -
                                                                      skip_threads_16781];
                            x_16717 = ((volatile __local
                                        double *) scan_arr_mem_16706)[local_tid_16698 -
                                                                      skip_threads_16781];
                            x_16718 = ((volatile __local
                                        double *) scan_arr_mem_16708)[local_tid_16698 -
                                                                      skip_threads_16781];
                            x_16719 = ((volatile __local
                                        double *) scan_arr_mem_16710)[local_tid_16698 -
                                                                      skip_threads_16781];
                        }
                        // perform operation
                        {
                            int32_t f_16725 = x_16715 | x_16720;
                            bool cond_16726 = slt32(0, x_16720);
                            double res_16727;
                            double res_16728;
                            double res_16729;
                            double res_16730;
                            
                            if (cond_16726) {
                                res_16727 = x_16721;
                                res_16728 = x_16722;
                                res_16729 = x_16723;
                                res_16730 = x_16724;
                            } else {
                                double y_16731 = x_16716 * x_16721;
                                double value_16732 = 1.0 / y_16731;
                                double y_16733 = x_16718 * x_16722;
                                double x_16734 = y_16731 + y_16733;
                                double res_16735 = value_16732 * x_16734;
                                double x_16736 = x_16717 * x_16721;
                                double y_16737 = x_16719 * x_16722;
                                double x_16738 = x_16736 + y_16737;
                                double res_16739 = value_16732 * x_16738;
                                double x_16740 = x_16716 * x_16723;
                                double y_16741 = x_16718 * x_16724;
                                double x_16742 = x_16740 + y_16741;
                                double res_16743 = value_16732 * x_16742;
                                double x_16744 = x_16717 * x_16723;
                                double y_16745 = x_16719 * x_16724;
                                double x_16746 = x_16744 + y_16745;
                                double res_16747 = value_16732 * x_16746;
                                
                                res_16727 = res_16735;
                                res_16728 = res_16739;
                                res_16729 = res_16743;
                                res_16730 = res_16747;
                            }
                            x_16715 = f_16725;
                            x_16716 = res_16727;
                            x_16717 = res_16728;
                            x_16718 = res_16729;
                            x_16719 = res_16730;
                        }
                    }
                    if (sle32(wave_sizze_16700, skip_threads_16781)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_16781, local_tid_16698 -
                              squot32(local_tid_16698, 32) * 32) &&
                        slt32(local_tid_16698, segscan_group_sizze_14150)) {
                        // write result
                        {
                            ((volatile __local
                              int32_t *) scan_arr_mem_16702)[local_tid_16698] =
                                x_16715;
                            x_16720 = x_16715;
                            ((volatile __local
                              double *) scan_arr_mem_16704)[local_tid_16698] =
                                x_16716;
                            x_16721 = x_16716;
                            ((volatile __local
                              double *) scan_arr_mem_16706)[local_tid_16698] =
                                x_16717;
                            x_16722 = x_16717;
                            ((volatile __local
                              double *) scan_arr_mem_16708)[local_tid_16698] =
                                x_16718;
                            x_16723 = x_16718;
                            ((volatile __local
                              double *) scan_arr_mem_16710)[local_tid_16698] =
                                x_16719;
                            x_16724 = x_16719;
                        }
                    }
                    if (sle32(wave_sizze_16700, skip_threads_16781)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_16781 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_16698 - squot32(local_tid_16698, 32) * 32) ==
                    31 && slt32(local_tid_16698, segscan_group_sizze_14150)) {
                    ((volatile __local
                      int32_t *) scan_arr_mem_16702)[squot32(local_tid_16698,
                                                             32)] = x_16715;
                    ((volatile __local
                      double *) scan_arr_mem_16704)[squot32(local_tid_16698,
                                                            32)] = x_16716;
                    ((volatile __local
                      double *) scan_arr_mem_16706)[squot32(local_tid_16698,
                                                            32)] = x_16717;
                    ((volatile __local
                      double *) scan_arr_mem_16708)[squot32(local_tid_16698,
                                                            32)] = x_16718;
                    ((volatile __local
                      double *) scan_arr_mem_16710)[squot32(local_tid_16698,
                                                            32)] = x_16719;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_16782;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_16698, 32) == 0 &&
                        slt32(local_tid_16698, segscan_group_sizze_14150)) {
                        x_16753 = ((volatile __local
                                    int32_t *) scan_arr_mem_16702)[local_tid_16698];
                        x_16754 = ((volatile __local
                                    double *) scan_arr_mem_16704)[local_tid_16698];
                        x_16755 = ((volatile __local
                                    double *) scan_arr_mem_16706)[local_tid_16698];
                        x_16756 = ((volatile __local
                                    double *) scan_arr_mem_16708)[local_tid_16698];
                        x_16757 = ((volatile __local
                                    double *) scan_arr_mem_16710)[local_tid_16698];
                        if ((local_tid_16698 - squot32(local_tid_16698, 32) *
                             32) == 0) {
                            x_16748 = x_16753;
                            x_16749 = x_16754;
                            x_16750 = x_16755;
                            x_16751 = x_16756;
                            x_16752 = x_16757;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_16782 = 1;
                    while (slt32(skip_threads_16782, 32)) {
                        if (sle32(skip_threads_16782, local_tid_16698 -
                                  squot32(local_tid_16698, 32) * 32) &&
                            (squot32(local_tid_16698, 32) == 0 &&
                             slt32(local_tid_16698,
                                   segscan_group_sizze_14150))) {
                            // read operands
                            {
                                x_16748 = ((volatile __local
                                            int32_t *) scan_arr_mem_16702)[local_tid_16698 -
                                                                           skip_threads_16782];
                                x_16749 = ((volatile __local
                                            double *) scan_arr_mem_16704)[local_tid_16698 -
                                                                          skip_threads_16782];
                                x_16750 = ((volatile __local
                                            double *) scan_arr_mem_16706)[local_tid_16698 -
                                                                          skip_threads_16782];
                                x_16751 = ((volatile __local
                                            double *) scan_arr_mem_16708)[local_tid_16698 -
                                                                          skip_threads_16782];
                                x_16752 = ((volatile __local
                                            double *) scan_arr_mem_16710)[local_tid_16698 -
                                                                          skip_threads_16782];
                            }
                            // perform operation
                            {
                                int32_t f_16758 = x_16748 | x_16753;
                                bool cond_16759 = slt32(0, x_16753);
                                double res_16760;
                                double res_16761;
                                double res_16762;
                                double res_16763;
                                
                                if (cond_16759) {
                                    res_16760 = x_16754;
                                    res_16761 = x_16755;
                                    res_16762 = x_16756;
                                    res_16763 = x_16757;
                                } else {
                                    double y_16764 = x_16749 * x_16754;
                                    double value_16765 = 1.0 / y_16764;
                                    double y_16766 = x_16751 * x_16755;
                                    double x_16767 = y_16764 + y_16766;
                                    double res_16768 = value_16765 * x_16767;
                                    double x_16769 = x_16750 * x_16754;
                                    double y_16770 = x_16752 * x_16755;
                                    double x_16771 = x_16769 + y_16770;
                                    double res_16772 = value_16765 * x_16771;
                                    double x_16773 = x_16749 * x_16756;
                                    double y_16774 = x_16751 * x_16757;
                                    double x_16775 = x_16773 + y_16774;
                                    double res_16776 = value_16765 * x_16775;
                                    double x_16777 = x_16750 * x_16756;
                                    double y_16778 = x_16752 * x_16757;
                                    double x_16779 = x_16777 + y_16778;
                                    double res_16780 = value_16765 * x_16779;
                                    
                                    res_16760 = res_16768;
                                    res_16761 = res_16772;
                                    res_16762 = res_16776;
                                    res_16763 = res_16780;
                                }
                                x_16748 = f_16758;
                                x_16749 = res_16760;
                                x_16750 = res_16761;
                                x_16751 = res_16762;
                                x_16752 = res_16763;
                            }
                        }
                        if (sle32(wave_sizze_16700, skip_threads_16782)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_16782, local_tid_16698 -
                                  squot32(local_tid_16698, 32) * 32) &&
                            (squot32(local_tid_16698, 32) == 0 &&
                             slt32(local_tid_16698,
                                   segscan_group_sizze_14150))) {
                            // write result
                            {
                                ((volatile __local
                                  int32_t *) scan_arr_mem_16702)[local_tid_16698] =
                                    x_16748;
                                x_16753 = x_16748;
                                ((volatile __local
                                  double *) scan_arr_mem_16704)[local_tid_16698] =
                                    x_16749;
                                x_16754 = x_16749;
                                ((volatile __local
                                  double *) scan_arr_mem_16706)[local_tid_16698] =
                                    x_16750;
                                x_16755 = x_16750;
                                ((volatile __local
                                  double *) scan_arr_mem_16708)[local_tid_16698] =
                                    x_16751;
                                x_16756 = x_16751;
                                ((volatile __local
                                  double *) scan_arr_mem_16710)[local_tid_16698] =
                                    x_16752;
                                x_16757 = x_16752;
                            }
                        }
                        if (sle32(wave_sizze_16700, skip_threads_16782)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_16782 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_16698, 32) == 0 ||
                      !slt32(local_tid_16698, segscan_group_sizze_14150))) {
                    // read operands
                    {
                        x_16720 = x_16715;
                        x_16721 = x_16716;
                        x_16722 = x_16717;
                        x_16723 = x_16718;
                        x_16724 = x_16719;
                        x_16715 = ((__local
                                    int32_t *) scan_arr_mem_16702)[squot32(local_tid_16698,
                                                                           32) -
                                                                   1];
                        x_16716 = ((__local
                                    double *) scan_arr_mem_16704)[squot32(local_tid_16698,
                                                                          32) -
                                                                  1];
                        x_16717 = ((__local
                                    double *) scan_arr_mem_16706)[squot32(local_tid_16698,
                                                                          32) -
                                                                  1];
                        x_16718 = ((__local
                                    double *) scan_arr_mem_16708)[squot32(local_tid_16698,
                                                                          32) -
                                                                  1];
                        x_16719 = ((__local
                                    double *) scan_arr_mem_16710)[squot32(local_tid_16698,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        int32_t f_16725 = x_16715 | x_16720;
                        bool cond_16726 = slt32(0, x_16720);
                        double res_16727;
                        double res_16728;
                        double res_16729;
                        double res_16730;
                        
                        if (cond_16726) {
                            res_16727 = x_16721;
                            res_16728 = x_16722;
                            res_16729 = x_16723;
                            res_16730 = x_16724;
                        } else {
                            double y_16731 = x_16716 * x_16721;
                            double value_16732 = 1.0 / y_16731;
                            double y_16733 = x_16718 * x_16722;
                            double x_16734 = y_16731 + y_16733;
                            double res_16735 = value_16732 * x_16734;
                            double x_16736 = x_16717 * x_16721;
                            double y_16737 = x_16719 * x_16722;
                            double x_16738 = x_16736 + y_16737;
                            double res_16739 = value_16732 * x_16738;
                            double x_16740 = x_16716 * x_16723;
                            double y_16741 = x_16718 * x_16724;
                            double x_16742 = x_16740 + y_16741;
                            double res_16743 = value_16732 * x_16742;
                            double x_16744 = x_16717 * x_16723;
                            double y_16745 = x_16719 * x_16724;
                            double x_16746 = x_16744 + y_16745;
                            double res_16747 = value_16732 * x_16746;
                            
                            res_16727 = res_16735;
                            res_16728 = res_16739;
                            res_16729 = res_16743;
                            res_16730 = res_16747;
                        }
                        x_16715 = f_16725;
                        x_16716 = res_16727;
                        x_16717 = res_16728;
                        x_16718 = res_16729;
                        x_16719 = res_16730;
                    }
                    // write final result
                    {
                        ((__local
                          int32_t *) scan_arr_mem_16702)[local_tid_16698] =
                            x_16715;
                        ((__local
                          double *) scan_arr_mem_16704)[local_tid_16698] =
                            x_16716;
                        ((__local
                          double *) scan_arr_mem_16706)[local_tid_16698] =
                            x_16717;
                        ((__local
                          double *) scan_arr_mem_16708)[local_tid_16698] =
                            x_16718;
                        ((__local
                          double *) scan_arr_mem_16710)[local_tid_16698] =
                            x_16719;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_16698, 32) == 0) {
                    ((__local int32_t *) scan_arr_mem_16702)[local_tid_16698] =
                        x_16720;
                    ((__local double *) scan_arr_mem_16704)[local_tid_16698] =
                        x_16721;
                    ((__local double *) scan_arr_mem_16706)[local_tid_16698] =
                        x_16722;
                    ((__local double *) scan_arr_mem_16708)[local_tid_16698] =
                        x_16723;
                    ((__local double *) scan_arr_mem_16710)[local_tid_16698] =
                        x_16724;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_14154, n_13426)) {
                    ((__global int32_t *) mem_16430)[gtid_14154] = ((__local
                                                                     int32_t *) scan_arr_mem_16702)[local_tid_16698];
                    ((__global double *) mem_16433)[gtid_14154] = ((__local
                                                                    double *) scan_arr_mem_16704)[local_tid_16698];
                    ((__global double *) mem_16436)[gtid_14154] = ((__local
                                                                    double *) scan_arr_mem_16706)[local_tid_16698];
                    ((__global double *) mem_16439)[gtid_14154] = ((__local
                                                                    double *) scan_arr_mem_16708)[local_tid_16698];
                    ((__global double *) mem_16442)[gtid_14154] = ((__local
                                                                    double *) scan_arr_mem_16710)[local_tid_16698];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_16783 = 0;
                bool should_load_carry_16784 = local_tid_16698 == 0 &&
                     !crosses_segment_16783;
                
                if (should_load_carry_16784) {
                    x_13504 = ((__local
                                int32_t *) scan_arr_mem_16702)[segscan_group_sizze_14150 -
                                                               1];
                    x_13505 = ((__local
                                double *) scan_arr_mem_16704)[segscan_group_sizze_14150 -
                                                              1];
                    x_13506 = ((__local
                                double *) scan_arr_mem_16706)[segscan_group_sizze_14150 -
                                                              1];
                    x_13507 = ((__local
                                double *) scan_arr_mem_16708)[segscan_group_sizze_14150 -
                                                              1];
                    x_13508 = ((__local
                                double *) scan_arr_mem_16710)[segscan_group_sizze_14150 -
                                                              1];
                }
                if (!should_load_carry_16784) {
                    x_13504 = 0;
                    x_13505 = 1.0;
                    x_13506 = 0.0;
                    x_13507 = 0.0;
                    x_13508 = 1.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_14150
}
__kernel void tridagParFlatziscan_stage1_14355(__global int *global_failure,
                                               __local volatile
                                               int64_t *scan_arr_mem_16869_backing_aligned_0,
                                               __local volatile
                                               int64_t *scan_arr_mem_16867_backing_aligned_1,
                                               int32_t n_13426, __global
                                               unsigned char *mem_16449,
                                               __global
                                               unsigned char *mem_16454,
                                               __global
                                               unsigned char *mem_16457,
                                               int32_t num_threads_16861)
{
    #define segscan_group_sizze_14350 (tridagParFlatzisegscan_group_sizze_14349)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16869_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16869_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16867_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16867_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16862;
    int32_t local_tid_16863;
    int32_t group_sizze_16866;
    int32_t wave_sizze_16865;
    int32_t group_tid_16864;
    
    global_tid_16862 = get_global_id(0);
    local_tid_16863 = get_local_id(0);
    group_sizze_16866 = get_local_size(0);
    wave_sizze_16865 = LOCKSTEP_WIDTH;
    group_tid_16864 = get_group_id(0);
    
    int32_t phys_tid_14355;
    
    phys_tid_14355 = global_tid_16862;
    
    __local char *scan_arr_mem_16867;
    __local char *scan_arr_mem_16869;
    
    scan_arr_mem_16867 = (__local char *) scan_arr_mem_16867_backing_0;
    scan_arr_mem_16869 = (__local char *) scan_arr_mem_16869_backing_1;
    
    int32_t x_13646;
    double x_13647;
    int32_t x_13648;
    double x_13649;
    
    x_13646 = 0;
    x_13647 = 0.0;
    for (int32_t j_16871 = 0; j_16871 < sdiv_up32(n_13426, num_threads_16861);
         j_16871++) {
        int32_t chunk_offset_16872 = segscan_group_sizze_14350 * j_16871 +
                group_tid_16864 * (segscan_group_sizze_14350 *
                                   sdiv_up32(n_13426, num_threads_16861));
        int32_t flat_idx_16873 = chunk_offset_16872 + local_tid_16863;
        int32_t gtid_14354 = flat_idx_16873;
        
        // threads in bounds read input
        {
            if (slt32(gtid_14354, n_13426)) {
                double x_13654 = ((__global double *) mem_16449)[gtid_14354];
                bool cond_13655 = 0.0 < x_13654;
                int32_t res_13656 = btoi_bool_i32(cond_13655);
                
                // write to-scan values to parameters
                {
                    x_13648 = res_13656;
                    x_13649 = x_13654;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt32(gtid_14354, n_13426)) {
                    x_13648 = 0;
                    x_13649 = 0.0;
                }
            }
            // combine with carry and write to local memory
            {
                int32_t f_13650 = x_13646 | x_13648;
                bool cond_13651 = slt32(0, x_13648);
                double res_13652;
                
                if (cond_13651) {
                    res_13652 = x_13649;
                } else {
                    double res_13653 = x_13647 + x_13649;
                    
                    res_13652 = res_13653;
                }
                ((__local int32_t *) scan_arr_mem_16867)[local_tid_16863] =
                    f_13650;
                ((__local double *) scan_arr_mem_16869)[local_tid_16863] =
                    res_13652;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t x_16874;
            double x_16875;
            int32_t x_16876;
            double x_16877;
            int32_t x_16882;
            double x_16883;
            int32_t x_16884;
            double x_16885;
            int32_t skip_threads_16890;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_16863, segscan_group_sizze_14350)) {
                    x_16876 = ((volatile __local
                                int32_t *) scan_arr_mem_16867)[local_tid_16863];
                    x_16877 = ((volatile __local
                                double *) scan_arr_mem_16869)[local_tid_16863];
                    if ((local_tid_16863 - squot32(local_tid_16863, 32) * 32) ==
                        0) {
                        x_16874 = x_16876;
                        x_16875 = x_16877;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_16890 = 1;
                while (slt32(skip_threads_16890, 32)) {
                    if (sle32(skip_threads_16890, local_tid_16863 -
                              squot32(local_tid_16863, 32) * 32) &&
                        slt32(local_tid_16863, segscan_group_sizze_14350)) {
                        // read operands
                        {
                            x_16874 = ((volatile __local
                                        int32_t *) scan_arr_mem_16867)[local_tid_16863 -
                                                                       skip_threads_16890];
                            x_16875 = ((volatile __local
                                        double *) scan_arr_mem_16869)[local_tid_16863 -
                                                                      skip_threads_16890];
                        }
                        // perform operation
                        {
                            int32_t f_16878 = x_16874 | x_16876;
                            bool cond_16879 = slt32(0, x_16876);
                            double res_16880;
                            
                            if (cond_16879) {
                                res_16880 = x_16877;
                            } else {
                                double res_16881 = x_16875 + x_16877;
                                
                                res_16880 = res_16881;
                            }
                            x_16874 = f_16878;
                            x_16875 = res_16880;
                        }
                    }
                    if (sle32(wave_sizze_16865, skip_threads_16890)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_16890, local_tid_16863 -
                              squot32(local_tid_16863, 32) * 32) &&
                        slt32(local_tid_16863, segscan_group_sizze_14350)) {
                        // write result
                        {
                            ((volatile __local
                              int32_t *) scan_arr_mem_16867)[local_tid_16863] =
                                x_16874;
                            x_16876 = x_16874;
                            ((volatile __local
                              double *) scan_arr_mem_16869)[local_tid_16863] =
                                x_16875;
                            x_16877 = x_16875;
                        }
                    }
                    if (sle32(wave_sizze_16865, skip_threads_16890)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_16890 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_16863 - squot32(local_tid_16863, 32) * 32) ==
                    31 && slt32(local_tid_16863, segscan_group_sizze_14350)) {
                    ((volatile __local
                      int32_t *) scan_arr_mem_16867)[squot32(local_tid_16863,
                                                             32)] = x_16874;
                    ((volatile __local
                      double *) scan_arr_mem_16869)[squot32(local_tid_16863,
                                                            32)] = x_16875;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_16891;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_16863, 32) == 0 &&
                        slt32(local_tid_16863, segscan_group_sizze_14350)) {
                        x_16884 = ((volatile __local
                                    int32_t *) scan_arr_mem_16867)[local_tid_16863];
                        x_16885 = ((volatile __local
                                    double *) scan_arr_mem_16869)[local_tid_16863];
                        if ((local_tid_16863 - squot32(local_tid_16863, 32) *
                             32) == 0) {
                            x_16882 = x_16884;
                            x_16883 = x_16885;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_16891 = 1;
                    while (slt32(skip_threads_16891, 32)) {
                        if (sle32(skip_threads_16891, local_tid_16863 -
                                  squot32(local_tid_16863, 32) * 32) &&
                            (squot32(local_tid_16863, 32) == 0 &&
                             slt32(local_tid_16863,
                                   segscan_group_sizze_14350))) {
                            // read operands
                            {
                                x_16882 = ((volatile __local
                                            int32_t *) scan_arr_mem_16867)[local_tid_16863 -
                                                                           skip_threads_16891];
                                x_16883 = ((volatile __local
                                            double *) scan_arr_mem_16869)[local_tid_16863 -
                                                                          skip_threads_16891];
                            }
                            // perform operation
                            {
                                int32_t f_16886 = x_16882 | x_16884;
                                bool cond_16887 = slt32(0, x_16884);
                                double res_16888;
                                
                                if (cond_16887) {
                                    res_16888 = x_16885;
                                } else {
                                    double res_16889 = x_16883 + x_16885;
                                    
                                    res_16888 = res_16889;
                                }
                                x_16882 = f_16886;
                                x_16883 = res_16888;
                            }
                        }
                        if (sle32(wave_sizze_16865, skip_threads_16891)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_16891, local_tid_16863 -
                                  squot32(local_tid_16863, 32) * 32) &&
                            (squot32(local_tid_16863, 32) == 0 &&
                             slt32(local_tid_16863,
                                   segscan_group_sizze_14350))) {
                            // write result
                            {
                                ((volatile __local
                                  int32_t *) scan_arr_mem_16867)[local_tid_16863] =
                                    x_16882;
                                x_16884 = x_16882;
                                ((volatile __local
                                  double *) scan_arr_mem_16869)[local_tid_16863] =
                                    x_16883;
                                x_16885 = x_16883;
                            }
                        }
                        if (sle32(wave_sizze_16865, skip_threads_16891)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_16891 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_16863, 32) == 0 ||
                      !slt32(local_tid_16863, segscan_group_sizze_14350))) {
                    // read operands
                    {
                        x_16876 = x_16874;
                        x_16877 = x_16875;
                        x_16874 = ((__local
                                    int32_t *) scan_arr_mem_16867)[squot32(local_tid_16863,
                                                                           32) -
                                                                   1];
                        x_16875 = ((__local
                                    double *) scan_arr_mem_16869)[squot32(local_tid_16863,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        int32_t f_16878 = x_16874 | x_16876;
                        bool cond_16879 = slt32(0, x_16876);
                        double res_16880;
                        
                        if (cond_16879) {
                            res_16880 = x_16877;
                        } else {
                            double res_16881 = x_16875 + x_16877;
                            
                            res_16880 = res_16881;
                        }
                        x_16874 = f_16878;
                        x_16875 = res_16880;
                    }
                    // write final result
                    {
                        ((__local
                          int32_t *) scan_arr_mem_16867)[local_tid_16863] =
                            x_16874;
                        ((__local
                          double *) scan_arr_mem_16869)[local_tid_16863] =
                            x_16875;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_16863, 32) == 0) {
                    ((__local int32_t *) scan_arr_mem_16867)[local_tid_16863] =
                        x_16876;
                    ((__local double *) scan_arr_mem_16869)[local_tid_16863] =
                        x_16877;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_14354, n_13426)) {
                    ((__global int32_t *) mem_16454)[gtid_14354] = ((__local
                                                                     int32_t *) scan_arr_mem_16867)[local_tid_16863];
                    ((__global double *) mem_16457)[gtid_14354] = ((__local
                                                                    double *) scan_arr_mem_16869)[local_tid_16863];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_16892 = 0;
                bool should_load_carry_16893 = local_tid_16863 == 0 &&
                     !crosses_segment_16892;
                
                if (should_load_carry_16893) {
                    x_13646 = ((__local
                                int32_t *) scan_arr_mem_16867)[segscan_group_sizze_14350 -
                                                               1];
                    x_13647 = ((__local
                                double *) scan_arr_mem_16869)[segscan_group_sizze_14350 -
                                                              1];
                }
                if (!should_load_carry_16893) {
                    x_13646 = 0;
                    x_13647 = 0.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_14350
}
__kernel void tridagParFlatziscan_stage1_14364(__global int *global_failure,
                                               __local volatile
                                               int64_t *scan_arr_mem_16939_backing_aligned_0,
                                               __local volatile
                                               int64_t *scan_arr_mem_16937_backing_aligned_1,
                                               __local volatile
                                               int64_t *scan_arr_mem_16935_backing_aligned_2,
                                               int32_t n_13426,
                                               int32_t segSizze_13434, __global
                                               unsigned char *a_mem_16409,
                                               __global
                                               unsigned char *y_mem_16412,
                                               __global
                                               unsigned char *mem_16415,
                                               __global
                                               unsigned char *mem_16446,
                                               __global
                                               unsigned char *mem_16461,
                                               __global
                                               unsigned char *mem_16464,
                                               __global
                                               unsigned char *mem_16467,
                                               int32_t num_threads_16929)
{
    #define segscan_group_sizze_14359 (tridagParFlatzisegscan_group_sizze_14358)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16939_backing_2 =
                          (__local volatile
                           char *) scan_arr_mem_16939_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16937_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16937_backing_aligned_1;
    __local volatile char *restrict scan_arr_mem_16935_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16935_backing_aligned_2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16930;
    int32_t local_tid_16931;
    int32_t group_sizze_16934;
    int32_t wave_sizze_16933;
    int32_t group_tid_16932;
    
    global_tid_16930 = get_global_id(0);
    local_tid_16931 = get_local_id(0);
    group_sizze_16934 = get_local_size(0);
    wave_sizze_16933 = LOCKSTEP_WIDTH;
    group_tid_16932 = get_group_id(0);
    
    int32_t phys_tid_14364;
    
    phys_tid_14364 = global_tid_16930;
    
    __local char *scan_arr_mem_16935;
    __local char *scan_arr_mem_16937;
    __local char *scan_arr_mem_16939;
    
    scan_arr_mem_16935 = (__local char *) scan_arr_mem_16935_backing_0;
    scan_arr_mem_16937 = (__local char *) scan_arr_mem_16937_backing_1;
    scan_arr_mem_16939 = (__local char *) scan_arr_mem_16939_backing_2;
    
    int32_t x_13670;
    double x_13671;
    double x_13672;
    int32_t x_13673;
    double x_13674;
    double x_13675;
    
    x_13670 = 0;
    x_13671 = 0.0;
    x_13672 = 1.0;
    for (int32_t j_16941 = 0; j_16941 < sdiv_up32(n_13426, num_threads_16929);
         j_16941++) {
        int32_t chunk_offset_16942 = segscan_group_sizze_14359 * j_16941 +
                group_tid_16932 * (segscan_group_sizze_14359 *
                                   sdiv_up32(n_13426, num_threads_16929));
        int32_t flat_idx_16943 = chunk_offset_16942 + local_tid_16931;
        int32_t gtid_14363 = flat_idx_16943;
        
        // threads in bounds read input
        {
            if (slt32(gtid_14363, n_13426)) {
                int32_t x_13686 = ((__global int32_t *) mem_16415)[gtid_14363];
                int32_t y_13687 = smod32(gtid_14363, segSizze_13434);
                bool cond_13688 = slt32(0, y_13687);
                double res_13689;
                
                if (cond_13688) {
                    double y_elem_13684 = ((__global
                                            double *) y_mem_16412)[gtid_14363];
                    
                    res_13689 = y_elem_13684;
                } else {
                    res_13689 = 0.0;
                }
                
                double res_13690;
                
                if (cond_13688) {
                    double a_elem_13685 = ((__global
                                            double *) a_mem_16409)[gtid_14363];
                    int32_t i_13691 = sub32(gtid_14363, 1);
                    double y_13692 = ((__global double *) mem_16446)[i_13691];
                    double y_13693 = a_elem_13685 / y_13692;
                    double res_13694 = 0.0 - y_13693;
                    
                    res_13690 = res_13694;
                } else {
                    res_13690 = 1.0;
                }
                // write to-scan values to parameters
                {
                    x_13673 = x_13686;
                    x_13674 = res_13689;
                    x_13675 = res_13690;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt32(gtid_14363, n_13426)) {
                    x_13673 = 0;
                    x_13674 = 0.0;
                    x_13675 = 1.0;
                }
            }
            // combine with carry and write to local memory
            {
                int32_t f_13676 = x_13670 | x_13673;
                bool cond_13677 = slt32(0, x_13673);
                double res_13678;
                double res_13679;
                
                if (cond_13677) {
                    res_13678 = x_13674;
                    res_13679 = x_13675;
                } else {
                    double y_13680 = x_13671 * x_13675;
                    double res_13681 = x_13674 + y_13680;
                    double res_13682 = x_13672 * x_13675;
                    
                    res_13678 = res_13681;
                    res_13679 = res_13682;
                }
                ((__local int32_t *) scan_arr_mem_16935)[local_tid_16931] =
                    f_13676;
                ((__local double *) scan_arr_mem_16937)[local_tid_16931] =
                    res_13678;
                ((__local double *) scan_arr_mem_16939)[local_tid_16931] =
                    res_13679;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t x_16944;
            double x_16945;
            double x_16946;
            int32_t x_16947;
            double x_16948;
            double x_16949;
            int32_t x_16957;
            double x_16958;
            double x_16959;
            int32_t x_16960;
            double x_16961;
            double x_16962;
            int32_t skip_threads_16970;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_16931, segscan_group_sizze_14359)) {
                    x_16947 = ((volatile __local
                                int32_t *) scan_arr_mem_16935)[local_tid_16931];
                    x_16948 = ((volatile __local
                                double *) scan_arr_mem_16937)[local_tid_16931];
                    x_16949 = ((volatile __local
                                double *) scan_arr_mem_16939)[local_tid_16931];
                    if ((local_tid_16931 - squot32(local_tid_16931, 32) * 32) ==
                        0) {
                        x_16944 = x_16947;
                        x_16945 = x_16948;
                        x_16946 = x_16949;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_16970 = 1;
                while (slt32(skip_threads_16970, 32)) {
                    if (sle32(skip_threads_16970, local_tid_16931 -
                              squot32(local_tid_16931, 32) * 32) &&
                        slt32(local_tid_16931, segscan_group_sizze_14359)) {
                        // read operands
                        {
                            x_16944 = ((volatile __local
                                        int32_t *) scan_arr_mem_16935)[local_tid_16931 -
                                                                       skip_threads_16970];
                            x_16945 = ((volatile __local
                                        double *) scan_arr_mem_16937)[local_tid_16931 -
                                                                      skip_threads_16970];
                            x_16946 = ((volatile __local
                                        double *) scan_arr_mem_16939)[local_tid_16931 -
                                                                      skip_threads_16970];
                        }
                        // perform operation
                        {
                            int32_t f_16950 = x_16944 | x_16947;
                            bool cond_16951 = slt32(0, x_16947);
                            double res_16952;
                            double res_16953;
                            
                            if (cond_16951) {
                                res_16952 = x_16948;
                                res_16953 = x_16949;
                            } else {
                                double y_16954 = x_16945 * x_16949;
                                double res_16955 = x_16948 + y_16954;
                                double res_16956 = x_16946 * x_16949;
                                
                                res_16952 = res_16955;
                                res_16953 = res_16956;
                            }
                            x_16944 = f_16950;
                            x_16945 = res_16952;
                            x_16946 = res_16953;
                        }
                    }
                    if (sle32(wave_sizze_16933, skip_threads_16970)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_16970, local_tid_16931 -
                              squot32(local_tid_16931, 32) * 32) &&
                        slt32(local_tid_16931, segscan_group_sizze_14359)) {
                        // write result
                        {
                            ((volatile __local
                              int32_t *) scan_arr_mem_16935)[local_tid_16931] =
                                x_16944;
                            x_16947 = x_16944;
                            ((volatile __local
                              double *) scan_arr_mem_16937)[local_tid_16931] =
                                x_16945;
                            x_16948 = x_16945;
                            ((volatile __local
                              double *) scan_arr_mem_16939)[local_tid_16931] =
                                x_16946;
                            x_16949 = x_16946;
                        }
                    }
                    if (sle32(wave_sizze_16933, skip_threads_16970)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_16970 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_16931 - squot32(local_tid_16931, 32) * 32) ==
                    31 && slt32(local_tid_16931, segscan_group_sizze_14359)) {
                    ((volatile __local
                      int32_t *) scan_arr_mem_16935)[squot32(local_tid_16931,
                                                             32)] = x_16944;
                    ((volatile __local
                      double *) scan_arr_mem_16937)[squot32(local_tid_16931,
                                                            32)] = x_16945;
                    ((volatile __local
                      double *) scan_arr_mem_16939)[squot32(local_tid_16931,
                                                            32)] = x_16946;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_16971;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_16931, 32) == 0 &&
                        slt32(local_tid_16931, segscan_group_sizze_14359)) {
                        x_16960 = ((volatile __local
                                    int32_t *) scan_arr_mem_16935)[local_tid_16931];
                        x_16961 = ((volatile __local
                                    double *) scan_arr_mem_16937)[local_tid_16931];
                        x_16962 = ((volatile __local
                                    double *) scan_arr_mem_16939)[local_tid_16931];
                        if ((local_tid_16931 - squot32(local_tid_16931, 32) *
                             32) == 0) {
                            x_16957 = x_16960;
                            x_16958 = x_16961;
                            x_16959 = x_16962;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_16971 = 1;
                    while (slt32(skip_threads_16971, 32)) {
                        if (sle32(skip_threads_16971, local_tid_16931 -
                                  squot32(local_tid_16931, 32) * 32) &&
                            (squot32(local_tid_16931, 32) == 0 &&
                             slt32(local_tid_16931,
                                   segscan_group_sizze_14359))) {
                            // read operands
                            {
                                x_16957 = ((volatile __local
                                            int32_t *) scan_arr_mem_16935)[local_tid_16931 -
                                                                           skip_threads_16971];
                                x_16958 = ((volatile __local
                                            double *) scan_arr_mem_16937)[local_tid_16931 -
                                                                          skip_threads_16971];
                                x_16959 = ((volatile __local
                                            double *) scan_arr_mem_16939)[local_tid_16931 -
                                                                          skip_threads_16971];
                            }
                            // perform operation
                            {
                                int32_t f_16963 = x_16957 | x_16960;
                                bool cond_16964 = slt32(0, x_16960);
                                double res_16965;
                                double res_16966;
                                
                                if (cond_16964) {
                                    res_16965 = x_16961;
                                    res_16966 = x_16962;
                                } else {
                                    double y_16967 = x_16958 * x_16962;
                                    double res_16968 = x_16961 + y_16967;
                                    double res_16969 = x_16959 * x_16962;
                                    
                                    res_16965 = res_16968;
                                    res_16966 = res_16969;
                                }
                                x_16957 = f_16963;
                                x_16958 = res_16965;
                                x_16959 = res_16966;
                            }
                        }
                        if (sle32(wave_sizze_16933, skip_threads_16971)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_16971, local_tid_16931 -
                                  squot32(local_tid_16931, 32) * 32) &&
                            (squot32(local_tid_16931, 32) == 0 &&
                             slt32(local_tid_16931,
                                   segscan_group_sizze_14359))) {
                            // write result
                            {
                                ((volatile __local
                                  int32_t *) scan_arr_mem_16935)[local_tid_16931] =
                                    x_16957;
                                x_16960 = x_16957;
                                ((volatile __local
                                  double *) scan_arr_mem_16937)[local_tid_16931] =
                                    x_16958;
                                x_16961 = x_16958;
                                ((volatile __local
                                  double *) scan_arr_mem_16939)[local_tid_16931] =
                                    x_16959;
                                x_16962 = x_16959;
                            }
                        }
                        if (sle32(wave_sizze_16933, skip_threads_16971)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_16971 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_16931, 32) == 0 ||
                      !slt32(local_tid_16931, segscan_group_sizze_14359))) {
                    // read operands
                    {
                        x_16947 = x_16944;
                        x_16948 = x_16945;
                        x_16949 = x_16946;
                        x_16944 = ((__local
                                    int32_t *) scan_arr_mem_16935)[squot32(local_tid_16931,
                                                                           32) -
                                                                   1];
                        x_16945 = ((__local
                                    double *) scan_arr_mem_16937)[squot32(local_tid_16931,
                                                                          32) -
                                                                  1];
                        x_16946 = ((__local
                                    double *) scan_arr_mem_16939)[squot32(local_tid_16931,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        int32_t f_16950 = x_16944 | x_16947;
                        bool cond_16951 = slt32(0, x_16947);
                        double res_16952;
                        double res_16953;
                        
                        if (cond_16951) {
                            res_16952 = x_16948;
                            res_16953 = x_16949;
                        } else {
                            double y_16954 = x_16945 * x_16949;
                            double res_16955 = x_16948 + y_16954;
                            double res_16956 = x_16946 * x_16949;
                            
                            res_16952 = res_16955;
                            res_16953 = res_16956;
                        }
                        x_16944 = f_16950;
                        x_16945 = res_16952;
                        x_16946 = res_16953;
                    }
                    // write final result
                    {
                        ((__local
                          int32_t *) scan_arr_mem_16935)[local_tid_16931] =
                            x_16944;
                        ((__local
                          double *) scan_arr_mem_16937)[local_tid_16931] =
                            x_16945;
                        ((__local
                          double *) scan_arr_mem_16939)[local_tid_16931] =
                            x_16946;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_16931, 32) == 0) {
                    ((__local int32_t *) scan_arr_mem_16935)[local_tid_16931] =
                        x_16947;
                    ((__local double *) scan_arr_mem_16937)[local_tid_16931] =
                        x_16948;
                    ((__local double *) scan_arr_mem_16939)[local_tid_16931] =
                        x_16949;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_14363, n_13426)) {
                    ((__global int32_t *) mem_16461)[gtid_14363] = ((__local
                                                                     int32_t *) scan_arr_mem_16935)[local_tid_16931];
                    ((__global double *) mem_16464)[gtid_14363] = ((__local
                                                                    double *) scan_arr_mem_16937)[local_tid_16931];
                    ((__global double *) mem_16467)[gtid_14363] = ((__local
                                                                    double *) scan_arr_mem_16939)[local_tid_16931];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_16972 = 0;
                bool should_load_carry_16973 = local_tid_16931 == 0 &&
                     !crosses_segment_16972;
                
                if (should_load_carry_16973) {
                    x_13670 = ((__local
                                int32_t *) scan_arr_mem_16935)[segscan_group_sizze_14359 -
                                                               1];
                    x_13671 = ((__local
                                double *) scan_arr_mem_16937)[segscan_group_sizze_14359 -
                                                              1];
                    x_13672 = ((__local
                                double *) scan_arr_mem_16939)[segscan_group_sizze_14359 -
                                                              1];
                }
                if (!should_load_carry_16973) {
                    x_13670 = 0;
                    x_13671 = 0.0;
                    x_13672 = 1.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_14359
}
__kernel void tridagParFlatziscan_stage1_14480(__global int *global_failure,
                                               __local volatile
                                               int64_t *scan_arr_mem_17034_backing_aligned_0,
                                               __local volatile
                                               int64_t *scan_arr_mem_17032_backing_aligned_1,
                                               int32_t n_13426, __global
                                               unsigned char *mem_16474,
                                               __global
                                               unsigned char *mem_16479,
                                               __global
                                               unsigned char *mem_16482,
                                               int32_t num_threads_17026)
{
    #define segscan_group_sizze_14475 (tridagParFlatzisegscan_group_sizze_14474)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17034_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17034_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17032_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17032_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17027;
    int32_t local_tid_17028;
    int32_t group_sizze_17031;
    int32_t wave_sizze_17030;
    int32_t group_tid_17029;
    
    global_tid_17027 = get_global_id(0);
    local_tid_17028 = get_local_id(0);
    group_sizze_17031 = get_local_size(0);
    wave_sizze_17030 = LOCKSTEP_WIDTH;
    group_tid_17029 = get_group_id(0);
    
    int32_t phys_tid_14480;
    
    phys_tid_14480 = global_tid_17027;
    
    __local char *scan_arr_mem_17032;
    __local char *scan_arr_mem_17034;
    
    scan_arr_mem_17032 = (__local char *) scan_arr_mem_17032_backing_0;
    scan_arr_mem_17034 = (__local char *) scan_arr_mem_17034_backing_1;
    
    int32_t x_13750;
    double x_13751;
    int32_t x_13752;
    double x_13753;
    
    x_13750 = 0;
    x_13751 = 0.0;
    for (int32_t j_17036 = 0; j_17036 < sdiv_up32(n_13426, num_threads_17026);
         j_17036++) {
        int32_t chunk_offset_17037 = segscan_group_sizze_14475 * j_17036 +
                group_tid_17029 * (segscan_group_sizze_14475 *
                                   sdiv_up32(n_13426, num_threads_17026));
        int32_t flat_idx_17038 = chunk_offset_17037 + local_tid_17028;
        int32_t gtid_14479 = flat_idx_17038;
        
        // threads in bounds read input
        {
            if (slt32(gtid_14479, n_13426)) {
                double x_13758 = ((__global double *) mem_16474)[gtid_14479];
                bool cond_13759 = 0.0 < x_13758;
                int32_t res_13760 = btoi_bool_i32(cond_13759);
                
                // write to-scan values to parameters
                {
                    x_13752 = res_13760;
                    x_13753 = x_13758;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt32(gtid_14479, n_13426)) {
                    x_13752 = 0;
                    x_13753 = 0.0;
                }
            }
            // combine with carry and write to local memory
            {
                int32_t f_13754 = x_13750 | x_13752;
                bool cond_13755 = slt32(0, x_13752);
                double res_13756;
                
                if (cond_13755) {
                    res_13756 = x_13753;
                } else {
                    double res_13757 = x_13751 + x_13753;
                    
                    res_13756 = res_13757;
                }
                ((__local int32_t *) scan_arr_mem_17032)[local_tid_17028] =
                    f_13754;
                ((__local double *) scan_arr_mem_17034)[local_tid_17028] =
                    res_13756;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t x_17039;
            double x_17040;
            int32_t x_17041;
            double x_17042;
            int32_t x_17047;
            double x_17048;
            int32_t x_17049;
            double x_17050;
            int32_t skip_threads_17055;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_17028, segscan_group_sizze_14475)) {
                    x_17041 = ((volatile __local
                                int32_t *) scan_arr_mem_17032)[local_tid_17028];
                    x_17042 = ((volatile __local
                                double *) scan_arr_mem_17034)[local_tid_17028];
                    if ((local_tid_17028 - squot32(local_tid_17028, 32) * 32) ==
                        0) {
                        x_17039 = x_17041;
                        x_17040 = x_17042;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_17055 = 1;
                while (slt32(skip_threads_17055, 32)) {
                    if (sle32(skip_threads_17055, local_tid_17028 -
                              squot32(local_tid_17028, 32) * 32) &&
                        slt32(local_tid_17028, segscan_group_sizze_14475)) {
                        // read operands
                        {
                            x_17039 = ((volatile __local
                                        int32_t *) scan_arr_mem_17032)[local_tid_17028 -
                                                                       skip_threads_17055];
                            x_17040 = ((volatile __local
                                        double *) scan_arr_mem_17034)[local_tid_17028 -
                                                                      skip_threads_17055];
                        }
                        // perform operation
                        {
                            int32_t f_17043 = x_17039 | x_17041;
                            bool cond_17044 = slt32(0, x_17041);
                            double res_17045;
                            
                            if (cond_17044) {
                                res_17045 = x_17042;
                            } else {
                                double res_17046 = x_17040 + x_17042;
                                
                                res_17045 = res_17046;
                            }
                            x_17039 = f_17043;
                            x_17040 = res_17045;
                        }
                    }
                    if (sle32(wave_sizze_17030, skip_threads_17055)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_17055, local_tid_17028 -
                              squot32(local_tid_17028, 32) * 32) &&
                        slt32(local_tid_17028, segscan_group_sizze_14475)) {
                        // write result
                        {
                            ((volatile __local
                              int32_t *) scan_arr_mem_17032)[local_tid_17028] =
                                x_17039;
                            x_17041 = x_17039;
                            ((volatile __local
                              double *) scan_arr_mem_17034)[local_tid_17028] =
                                x_17040;
                            x_17042 = x_17040;
                        }
                    }
                    if (sle32(wave_sizze_17030, skip_threads_17055)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_17055 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_17028 - squot32(local_tid_17028, 32) * 32) ==
                    31 && slt32(local_tid_17028, segscan_group_sizze_14475)) {
                    ((volatile __local
                      int32_t *) scan_arr_mem_17032)[squot32(local_tid_17028,
                                                             32)] = x_17039;
                    ((volatile __local
                      double *) scan_arr_mem_17034)[squot32(local_tid_17028,
                                                            32)] = x_17040;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_17056;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_17028, 32) == 0 &&
                        slt32(local_tid_17028, segscan_group_sizze_14475)) {
                        x_17049 = ((volatile __local
                                    int32_t *) scan_arr_mem_17032)[local_tid_17028];
                        x_17050 = ((volatile __local
                                    double *) scan_arr_mem_17034)[local_tid_17028];
                        if ((local_tid_17028 - squot32(local_tid_17028, 32) *
                             32) == 0) {
                            x_17047 = x_17049;
                            x_17048 = x_17050;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_17056 = 1;
                    while (slt32(skip_threads_17056, 32)) {
                        if (sle32(skip_threads_17056, local_tid_17028 -
                                  squot32(local_tid_17028, 32) * 32) &&
                            (squot32(local_tid_17028, 32) == 0 &&
                             slt32(local_tid_17028,
                                   segscan_group_sizze_14475))) {
                            // read operands
                            {
                                x_17047 = ((volatile __local
                                            int32_t *) scan_arr_mem_17032)[local_tid_17028 -
                                                                           skip_threads_17056];
                                x_17048 = ((volatile __local
                                            double *) scan_arr_mem_17034)[local_tid_17028 -
                                                                          skip_threads_17056];
                            }
                            // perform operation
                            {
                                int32_t f_17051 = x_17047 | x_17049;
                                bool cond_17052 = slt32(0, x_17049);
                                double res_17053;
                                
                                if (cond_17052) {
                                    res_17053 = x_17050;
                                } else {
                                    double res_17054 = x_17048 + x_17050;
                                    
                                    res_17053 = res_17054;
                                }
                                x_17047 = f_17051;
                                x_17048 = res_17053;
                            }
                        }
                        if (sle32(wave_sizze_17030, skip_threads_17056)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_17056, local_tid_17028 -
                                  squot32(local_tid_17028, 32) * 32) &&
                            (squot32(local_tid_17028, 32) == 0 &&
                             slt32(local_tid_17028,
                                   segscan_group_sizze_14475))) {
                            // write result
                            {
                                ((volatile __local
                                  int32_t *) scan_arr_mem_17032)[local_tid_17028] =
                                    x_17047;
                                x_17049 = x_17047;
                                ((volatile __local
                                  double *) scan_arr_mem_17034)[local_tid_17028] =
                                    x_17048;
                                x_17050 = x_17048;
                            }
                        }
                        if (sle32(wave_sizze_17030, skip_threads_17056)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_17056 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_17028, 32) == 0 ||
                      !slt32(local_tid_17028, segscan_group_sizze_14475))) {
                    // read operands
                    {
                        x_17041 = x_17039;
                        x_17042 = x_17040;
                        x_17039 = ((__local
                                    int32_t *) scan_arr_mem_17032)[squot32(local_tid_17028,
                                                                           32) -
                                                                   1];
                        x_17040 = ((__local
                                    double *) scan_arr_mem_17034)[squot32(local_tid_17028,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        int32_t f_17043 = x_17039 | x_17041;
                        bool cond_17044 = slt32(0, x_17041);
                        double res_17045;
                        
                        if (cond_17044) {
                            res_17045 = x_17042;
                        } else {
                            double res_17046 = x_17040 + x_17042;
                            
                            res_17045 = res_17046;
                        }
                        x_17039 = f_17043;
                        x_17040 = res_17045;
                    }
                    // write final result
                    {
                        ((__local
                          int32_t *) scan_arr_mem_17032)[local_tid_17028] =
                            x_17039;
                        ((__local
                          double *) scan_arr_mem_17034)[local_tid_17028] =
                            x_17040;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_17028, 32) == 0) {
                    ((__local int32_t *) scan_arr_mem_17032)[local_tid_17028] =
                        x_17041;
                    ((__local double *) scan_arr_mem_17034)[local_tid_17028] =
                        x_17042;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_14479, n_13426)) {
                    ((__global int32_t *) mem_16479)[gtid_14479] = ((__local
                                                                     int32_t *) scan_arr_mem_17032)[local_tid_17028];
                    ((__global double *) mem_16482)[gtid_14479] = ((__local
                                                                    double *) scan_arr_mem_17034)[local_tid_17028];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_17057 = 0;
                bool should_load_carry_17058 = local_tid_17028 == 0 &&
                     !crosses_segment_17057;
                
                if (should_load_carry_17058) {
                    x_13750 = ((__local
                                int32_t *) scan_arr_mem_17032)[segscan_group_sizze_14475 -
                                                               1];
                    x_13751 = ((__local
                                double *) scan_arr_mem_17034)[segscan_group_sizze_14475 -
                                                              1];
                }
                if (!should_load_carry_17058) {
                    x_13750 = 0;
                    x_13751 = 0.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_14475
}
__kernel void tridagParFlatziscan_stage1_14489(__global int *global_failure,
                                               __local volatile
                                               int64_t *scan_arr_mem_17104_backing_aligned_0,
                                               __local volatile
                                               int64_t *scan_arr_mem_17102_backing_aligned_1,
                                               __local volatile
                                               int64_t *scan_arr_mem_17100_backing_aligned_2,
                                               int32_t n_13426,
                                               int32_t segSizze_13434, __global
                                               unsigned char *c_mem_16411,
                                               __global
                                               unsigned char *mem_16415,
                                               __global
                                               unsigned char *mem_16446,
                                               __global
                                               unsigned char *mem_16471,
                                               __global
                                               unsigned char *mem_16486,
                                               __global
                                               unsigned char *mem_16489,
                                               __global
                                               unsigned char *mem_16492,
                                               int32_t num_threads_17094)
{
    #define segscan_group_sizze_14484 (tridagParFlatzisegscan_group_sizze_14483)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17104_backing_2 =
                          (__local volatile
                           char *) scan_arr_mem_17104_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17102_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17102_backing_aligned_1;
    __local volatile char *restrict scan_arr_mem_17100_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17100_backing_aligned_2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17095;
    int32_t local_tid_17096;
    int32_t group_sizze_17099;
    int32_t wave_sizze_17098;
    int32_t group_tid_17097;
    
    global_tid_17095 = get_global_id(0);
    local_tid_17096 = get_local_id(0);
    group_sizze_17099 = get_local_size(0);
    wave_sizze_17098 = LOCKSTEP_WIDTH;
    group_tid_17097 = get_group_id(0);
    
    int32_t phys_tid_14489;
    
    phys_tid_14489 = global_tid_17095;
    
    __local char *scan_arr_mem_17100;
    __local char *scan_arr_mem_17102;
    __local char *scan_arr_mem_17104;
    
    scan_arr_mem_17100 = (__local char *) scan_arr_mem_17100_backing_0;
    scan_arr_mem_17102 = (__local char *) scan_arr_mem_17102_backing_1;
    scan_arr_mem_17104 = (__local char *) scan_arr_mem_17104_backing_2;
    
    int32_t x_13774;
    double x_13775;
    double x_13776;
    int32_t x_13777;
    double x_13778;
    double x_13779;
    
    x_13774 = 0;
    x_13775 = 0.0;
    x_13776 = 1.0;
    for (int32_t j_17106 = 0; j_17106 < sdiv_up32(n_13426, num_threads_17094);
         j_17106++) {
        int32_t chunk_offset_17107 = segscan_group_sizze_14484 * j_17106 +
                group_tid_17097 * (segscan_group_sizze_14484 *
                                   sdiv_up32(n_13426, num_threads_17094));
        int32_t flat_idx_17108 = chunk_offset_17107 + local_tid_17096;
        int32_t gtid_14488 = flat_idx_17108;
        
        // threads in bounds read input
        {
            if (slt32(gtid_14488, n_13426)) {
                int32_t x_13788 = ((__global int32_t *) mem_16415)[gtid_14488];
                int32_t seg_13789 = sdiv32(gtid_14488, segSizze_13434);
                int32_t segInd_13790 = smod32(gtid_14488, segSizze_13434);
                int32_t x_13791 = mul32(segSizze_13434, seg_13789);
                int32_t x_13792 = add32(segSizze_13434, x_13791);
                int32_t x_13793 = sub32(x_13792, segInd_13790);
                int32_t i_13794 = sub32(x_13793, 1);
                bool cond_13795 = slt32(0, i_13794);
                double res_13796;
                double res_13797;
                
                if (cond_13795) {
                    double x_13798 = ((__global double *) mem_16471)[i_13794];
                    double y_13799 = ((__global double *) mem_16446)[i_13794];
                    double res_13800 = x_13798 / y_13799;
                    double x_13801 = ((__global double *) c_mem_16411)[i_13794];
                    double y_13802 = x_13801 / y_13799;
                    double res_13803 = 0.0 - y_13802;
                    
                    res_13796 = res_13800;
                    res_13797 = res_13803;
                } else {
                    res_13796 = 0.0;
                    res_13797 = 1.0;
                }
                // write to-scan values to parameters
                {
                    x_13777 = x_13788;
                    x_13778 = res_13796;
                    x_13779 = res_13797;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt32(gtid_14488, n_13426)) {
                    x_13777 = 0;
                    x_13778 = 0.0;
                    x_13779 = 1.0;
                }
            }
            // combine with carry and write to local memory
            {
                int32_t f_13780 = x_13774 | x_13777;
                bool cond_13781 = slt32(0, x_13777);
                double res_13782;
                double res_13783;
                
                if (cond_13781) {
                    res_13782 = x_13778;
                    res_13783 = x_13779;
                } else {
                    double y_13784 = x_13775 * x_13779;
                    double res_13785 = x_13778 + y_13784;
                    double res_13786 = x_13776 * x_13779;
                    
                    res_13782 = res_13785;
                    res_13783 = res_13786;
                }
                ((__local int32_t *) scan_arr_mem_17100)[local_tid_17096] =
                    f_13780;
                ((__local double *) scan_arr_mem_17102)[local_tid_17096] =
                    res_13782;
                ((__local double *) scan_arr_mem_17104)[local_tid_17096] =
                    res_13783;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t x_17109;
            double x_17110;
            double x_17111;
            int32_t x_17112;
            double x_17113;
            double x_17114;
            int32_t x_17122;
            double x_17123;
            double x_17124;
            int32_t x_17125;
            double x_17126;
            double x_17127;
            int32_t skip_threads_17135;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_17096, segscan_group_sizze_14484)) {
                    x_17112 = ((volatile __local
                                int32_t *) scan_arr_mem_17100)[local_tid_17096];
                    x_17113 = ((volatile __local
                                double *) scan_arr_mem_17102)[local_tid_17096];
                    x_17114 = ((volatile __local
                                double *) scan_arr_mem_17104)[local_tid_17096];
                    if ((local_tid_17096 - squot32(local_tid_17096, 32) * 32) ==
                        0) {
                        x_17109 = x_17112;
                        x_17110 = x_17113;
                        x_17111 = x_17114;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_17135 = 1;
                while (slt32(skip_threads_17135, 32)) {
                    if (sle32(skip_threads_17135, local_tid_17096 -
                              squot32(local_tid_17096, 32) * 32) &&
                        slt32(local_tid_17096, segscan_group_sizze_14484)) {
                        // read operands
                        {
                            x_17109 = ((volatile __local
                                        int32_t *) scan_arr_mem_17100)[local_tid_17096 -
                                                                       skip_threads_17135];
                            x_17110 = ((volatile __local
                                        double *) scan_arr_mem_17102)[local_tid_17096 -
                                                                      skip_threads_17135];
                            x_17111 = ((volatile __local
                                        double *) scan_arr_mem_17104)[local_tid_17096 -
                                                                      skip_threads_17135];
                        }
                        // perform operation
                        {
                            int32_t f_17115 = x_17109 | x_17112;
                            bool cond_17116 = slt32(0, x_17112);
                            double res_17117;
                            double res_17118;
                            
                            if (cond_17116) {
                                res_17117 = x_17113;
                                res_17118 = x_17114;
                            } else {
                                double y_17119 = x_17110 * x_17114;
                                double res_17120 = x_17113 + y_17119;
                                double res_17121 = x_17111 * x_17114;
                                
                                res_17117 = res_17120;
                                res_17118 = res_17121;
                            }
                            x_17109 = f_17115;
                            x_17110 = res_17117;
                            x_17111 = res_17118;
                        }
                    }
                    if (sle32(wave_sizze_17098, skip_threads_17135)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_17135, local_tid_17096 -
                              squot32(local_tid_17096, 32) * 32) &&
                        slt32(local_tid_17096, segscan_group_sizze_14484)) {
                        // write result
                        {
                            ((volatile __local
                              int32_t *) scan_arr_mem_17100)[local_tid_17096] =
                                x_17109;
                            x_17112 = x_17109;
                            ((volatile __local
                              double *) scan_arr_mem_17102)[local_tid_17096] =
                                x_17110;
                            x_17113 = x_17110;
                            ((volatile __local
                              double *) scan_arr_mem_17104)[local_tid_17096] =
                                x_17111;
                            x_17114 = x_17111;
                        }
                    }
                    if (sle32(wave_sizze_17098, skip_threads_17135)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_17135 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_17096 - squot32(local_tid_17096, 32) * 32) ==
                    31 && slt32(local_tid_17096, segscan_group_sizze_14484)) {
                    ((volatile __local
                      int32_t *) scan_arr_mem_17100)[squot32(local_tid_17096,
                                                             32)] = x_17109;
                    ((volatile __local
                      double *) scan_arr_mem_17102)[squot32(local_tid_17096,
                                                            32)] = x_17110;
                    ((volatile __local
                      double *) scan_arr_mem_17104)[squot32(local_tid_17096,
                                                            32)] = x_17111;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_17136;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_17096, 32) == 0 &&
                        slt32(local_tid_17096, segscan_group_sizze_14484)) {
                        x_17125 = ((volatile __local
                                    int32_t *) scan_arr_mem_17100)[local_tid_17096];
                        x_17126 = ((volatile __local
                                    double *) scan_arr_mem_17102)[local_tid_17096];
                        x_17127 = ((volatile __local
                                    double *) scan_arr_mem_17104)[local_tid_17096];
                        if ((local_tid_17096 - squot32(local_tid_17096, 32) *
                             32) == 0) {
                            x_17122 = x_17125;
                            x_17123 = x_17126;
                            x_17124 = x_17127;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_17136 = 1;
                    while (slt32(skip_threads_17136, 32)) {
                        if (sle32(skip_threads_17136, local_tid_17096 -
                                  squot32(local_tid_17096, 32) * 32) &&
                            (squot32(local_tid_17096, 32) == 0 &&
                             slt32(local_tid_17096,
                                   segscan_group_sizze_14484))) {
                            // read operands
                            {
                                x_17122 = ((volatile __local
                                            int32_t *) scan_arr_mem_17100)[local_tid_17096 -
                                                                           skip_threads_17136];
                                x_17123 = ((volatile __local
                                            double *) scan_arr_mem_17102)[local_tid_17096 -
                                                                          skip_threads_17136];
                                x_17124 = ((volatile __local
                                            double *) scan_arr_mem_17104)[local_tid_17096 -
                                                                          skip_threads_17136];
                            }
                            // perform operation
                            {
                                int32_t f_17128 = x_17122 | x_17125;
                                bool cond_17129 = slt32(0, x_17125);
                                double res_17130;
                                double res_17131;
                                
                                if (cond_17129) {
                                    res_17130 = x_17126;
                                    res_17131 = x_17127;
                                } else {
                                    double y_17132 = x_17123 * x_17127;
                                    double res_17133 = x_17126 + y_17132;
                                    double res_17134 = x_17124 * x_17127;
                                    
                                    res_17130 = res_17133;
                                    res_17131 = res_17134;
                                }
                                x_17122 = f_17128;
                                x_17123 = res_17130;
                                x_17124 = res_17131;
                            }
                        }
                        if (sle32(wave_sizze_17098, skip_threads_17136)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_17136, local_tid_17096 -
                                  squot32(local_tid_17096, 32) * 32) &&
                            (squot32(local_tid_17096, 32) == 0 &&
                             slt32(local_tid_17096,
                                   segscan_group_sizze_14484))) {
                            // write result
                            {
                                ((volatile __local
                                  int32_t *) scan_arr_mem_17100)[local_tid_17096] =
                                    x_17122;
                                x_17125 = x_17122;
                                ((volatile __local
                                  double *) scan_arr_mem_17102)[local_tid_17096] =
                                    x_17123;
                                x_17126 = x_17123;
                                ((volatile __local
                                  double *) scan_arr_mem_17104)[local_tid_17096] =
                                    x_17124;
                                x_17127 = x_17124;
                            }
                        }
                        if (sle32(wave_sizze_17098, skip_threads_17136)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_17136 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_17096, 32) == 0 ||
                      !slt32(local_tid_17096, segscan_group_sizze_14484))) {
                    // read operands
                    {
                        x_17112 = x_17109;
                        x_17113 = x_17110;
                        x_17114 = x_17111;
                        x_17109 = ((__local
                                    int32_t *) scan_arr_mem_17100)[squot32(local_tid_17096,
                                                                           32) -
                                                                   1];
                        x_17110 = ((__local
                                    double *) scan_arr_mem_17102)[squot32(local_tid_17096,
                                                                          32) -
                                                                  1];
                        x_17111 = ((__local
                                    double *) scan_arr_mem_17104)[squot32(local_tid_17096,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        int32_t f_17115 = x_17109 | x_17112;
                        bool cond_17116 = slt32(0, x_17112);
                        double res_17117;
                        double res_17118;
                        
                        if (cond_17116) {
                            res_17117 = x_17113;
                            res_17118 = x_17114;
                        } else {
                            double y_17119 = x_17110 * x_17114;
                            double res_17120 = x_17113 + y_17119;
                            double res_17121 = x_17111 * x_17114;
                            
                            res_17117 = res_17120;
                            res_17118 = res_17121;
                        }
                        x_17109 = f_17115;
                        x_17110 = res_17117;
                        x_17111 = res_17118;
                    }
                    // write final result
                    {
                        ((__local
                          int32_t *) scan_arr_mem_17100)[local_tid_17096] =
                            x_17109;
                        ((__local
                          double *) scan_arr_mem_17102)[local_tid_17096] =
                            x_17110;
                        ((__local
                          double *) scan_arr_mem_17104)[local_tid_17096] =
                            x_17111;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_17096, 32) == 0) {
                    ((__local int32_t *) scan_arr_mem_17100)[local_tid_17096] =
                        x_17112;
                    ((__local double *) scan_arr_mem_17102)[local_tid_17096] =
                        x_17113;
                    ((__local double *) scan_arr_mem_17104)[local_tid_17096] =
                        x_17114;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_14488, n_13426)) {
                    ((__global int32_t *) mem_16486)[gtid_14488] = ((__local
                                                                     int32_t *) scan_arr_mem_17100)[local_tid_17096];
                    ((__global double *) mem_16489)[gtid_14488] = ((__local
                                                                    double *) scan_arr_mem_17102)[local_tid_17096];
                    ((__global double *) mem_16492)[gtid_14488] = ((__local
                                                                    double *) scan_arr_mem_17104)[local_tid_17096];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_17137 = 0;
                bool should_load_carry_17138 = local_tid_17096 == 0 &&
                     !crosses_segment_17137;
                
                if (should_load_carry_17138) {
                    x_13774 = ((__local
                                int32_t *) scan_arr_mem_17100)[segscan_group_sizze_14484 -
                                                               1];
                    x_13775 = ((__local
                                double *) scan_arr_mem_17102)[segscan_group_sizze_14484 -
                                                              1];
                    x_13776 = ((__local
                                double *) scan_arr_mem_17104)[segscan_group_sizze_14484 -
                                                              1];
                }
                if (!should_load_carry_17138) {
                    x_13774 = 0;
                    x_13775 = 0.0;
                    x_13776 = 1.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_14484
}
__kernel void tridagParFlatziscan_stage2_14146(__global int *global_failure,
                                               __local volatile
                                               int64_t *scan_arr_mem_16668_backing_aligned_0,
                                               __local volatile
                                               int64_t *scan_arr_mem_16666_backing_aligned_1,
                                               int32_t n_13426, __global
                                               unsigned char *mem_16423,
                                               __global
                                               unsigned char *mem_16426,
                                               int32_t stage1_num_groups_16627,
                                               int32_t num_threads_16628)
{
    #define segscan_group_sizze_14141 (tridagParFlatzisegscan_group_sizze_14140)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16668_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16668_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16666_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16666_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16661;
    int32_t local_tid_16662;
    int32_t group_sizze_16665;
    int32_t wave_sizze_16664;
    int32_t group_tid_16663;
    
    global_tid_16661 = get_global_id(0);
    local_tid_16662 = get_local_id(0);
    group_sizze_16665 = get_local_size(0);
    wave_sizze_16664 = LOCKSTEP_WIDTH;
    group_tid_16663 = get_group_id(0);
    
    int32_t phys_tid_14146;
    
    phys_tid_14146 = global_tid_16661;
    
    __local char *scan_arr_mem_16666;
    __local char *scan_arr_mem_16668;
    
    scan_arr_mem_16666 = (__local char *) scan_arr_mem_16666_backing_0;
    scan_arr_mem_16668 = (__local char *) scan_arr_mem_16668_backing_1;
    
    int32_t flat_idx_16670;
    
    flat_idx_16670 = (local_tid_16662 + 1) * (segscan_group_sizze_14141 *
                                              sdiv_up32(n_13426,
                                                        num_threads_16628)) - 1;
    
    int32_t gtid_14145;
    
    gtid_14145 = flat_idx_16670;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_14145, n_13426)) {
            ((__local int32_t *) scan_arr_mem_16666)[local_tid_16662] =
                ((__global int32_t *) mem_16423)[gtid_14145];
            ((__local double *) scan_arr_mem_16668)[local_tid_16662] =
                ((__global double *) mem_16426)[gtid_14145];
        } else {
            ((__local int32_t *) scan_arr_mem_16666)[local_tid_16662] = 0;
            ((__local double *) scan_arr_mem_16668)[local_tid_16662] = 0.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t x_13478;
    double x_13479;
    int32_t x_13480;
    double x_13481;
    int32_t x_16671;
    double x_16672;
    int32_t x_16673;
    double x_16674;
    int32_t skip_threads_16679;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16662, stage1_num_groups_16627)) {
            x_13480 = ((volatile __local
                        int32_t *) scan_arr_mem_16666)[local_tid_16662];
            x_13481 = ((volatile __local
                        double *) scan_arr_mem_16668)[local_tid_16662];
            if ((local_tid_16662 - squot32(local_tid_16662, 32) * 32) == 0) {
                x_13478 = x_13480;
                x_13479 = x_13481;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16679 = 1;
        while (slt32(skip_threads_16679, 32)) {
            if (sle32(skip_threads_16679, local_tid_16662 -
                      squot32(local_tid_16662, 32) * 32) &&
                slt32(local_tid_16662, stage1_num_groups_16627)) {
                // read operands
                {
                    x_13478 = ((volatile __local
                                int32_t *) scan_arr_mem_16666)[local_tid_16662 -
                                                               skip_threads_16679];
                    x_13479 = ((volatile __local
                                double *) scan_arr_mem_16668)[local_tid_16662 -
                                                              skip_threads_16679];
                }
                // perform operation
                {
                    int32_t f_13482 = x_13478 | x_13480;
                    bool cond_13483 = slt32(0, x_13480);
                    double res_13484;
                    
                    if (cond_13483) {
                        res_13484 = x_13481;
                    } else {
                        double res_13485 = x_13479 + x_13481;
                        
                        res_13484 = res_13485;
                    }
                    x_13478 = f_13482;
                    x_13479 = res_13484;
                }
            }
            if (sle32(wave_sizze_16664, skip_threads_16679)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16679, local_tid_16662 -
                      squot32(local_tid_16662, 32) * 32) &&
                slt32(local_tid_16662, stage1_num_groups_16627)) {
                // write result
                {
                    ((volatile __local
                      int32_t *) scan_arr_mem_16666)[local_tid_16662] = x_13478;
                    x_13480 = x_13478;
                    ((volatile __local
                      double *) scan_arr_mem_16668)[local_tid_16662] = x_13479;
                    x_13481 = x_13479;
                }
            }
            if (sle32(wave_sizze_16664, skip_threads_16679)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16679 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16662 - squot32(local_tid_16662, 32) * 32) == 31 &&
            slt32(local_tid_16662, stage1_num_groups_16627)) {
            ((volatile __local
              int32_t *) scan_arr_mem_16666)[squot32(local_tid_16662, 32)] =
                x_13478;
            ((volatile __local
              double *) scan_arr_mem_16668)[squot32(local_tid_16662, 32)] =
                x_13479;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16680;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16662, 32) == 0 && slt32(local_tid_16662,
                                                           stage1_num_groups_16627)) {
                x_16673 = ((volatile __local
                            int32_t *) scan_arr_mem_16666)[local_tid_16662];
                x_16674 = ((volatile __local
                            double *) scan_arr_mem_16668)[local_tid_16662];
                if ((local_tid_16662 - squot32(local_tid_16662, 32) * 32) ==
                    0) {
                    x_16671 = x_16673;
                    x_16672 = x_16674;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16680 = 1;
            while (slt32(skip_threads_16680, 32)) {
                if (sle32(skip_threads_16680, local_tid_16662 -
                          squot32(local_tid_16662, 32) * 32) &&
                    (squot32(local_tid_16662, 32) == 0 && slt32(local_tid_16662,
                                                                stage1_num_groups_16627))) {
                    // read operands
                    {
                        x_16671 = ((volatile __local
                                    int32_t *) scan_arr_mem_16666)[local_tid_16662 -
                                                                   skip_threads_16680];
                        x_16672 = ((volatile __local
                                    double *) scan_arr_mem_16668)[local_tid_16662 -
                                                                  skip_threads_16680];
                    }
                    // perform operation
                    {
                        int32_t f_16675 = x_16671 | x_16673;
                        bool cond_16676 = slt32(0, x_16673);
                        double res_16677;
                        
                        if (cond_16676) {
                            res_16677 = x_16674;
                        } else {
                            double res_16678 = x_16672 + x_16674;
                            
                            res_16677 = res_16678;
                        }
                        x_16671 = f_16675;
                        x_16672 = res_16677;
                    }
                }
                if (sle32(wave_sizze_16664, skip_threads_16680)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16680, local_tid_16662 -
                          squot32(local_tid_16662, 32) * 32) &&
                    (squot32(local_tid_16662, 32) == 0 && slt32(local_tid_16662,
                                                                stage1_num_groups_16627))) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) scan_arr_mem_16666)[local_tid_16662] =
                            x_16671;
                        x_16673 = x_16671;
                        ((volatile __local
                          double *) scan_arr_mem_16668)[local_tid_16662] =
                            x_16672;
                        x_16674 = x_16672;
                    }
                }
                if (sle32(wave_sizze_16664, skip_threads_16680)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16680 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16662, 32) == 0 || !slt32(local_tid_16662,
                                                          stage1_num_groups_16627))) {
            // read operands
            {
                x_13480 = x_13478;
                x_13481 = x_13479;
                x_13478 = ((__local
                            int32_t *) scan_arr_mem_16666)[squot32(local_tid_16662,
                                                                   32) - 1];
                x_13479 = ((__local
                            double *) scan_arr_mem_16668)[squot32(local_tid_16662,
                                                                  32) - 1];
            }
            // perform operation
            {
                int32_t f_13482 = x_13478 | x_13480;
                bool cond_13483 = slt32(0, x_13480);
                double res_13484;
                
                if (cond_13483) {
                    res_13484 = x_13481;
                } else {
                    double res_13485 = x_13479 + x_13481;
                    
                    res_13484 = res_13485;
                }
                x_13478 = f_13482;
                x_13479 = res_13484;
            }
            // write final result
            {
                ((__local int32_t *) scan_arr_mem_16666)[local_tid_16662] =
                    x_13478;
                ((__local double *) scan_arr_mem_16668)[local_tid_16662] =
                    x_13479;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16662, 32) == 0) {
            ((__local int32_t *) scan_arr_mem_16666)[local_tid_16662] = x_13480;
            ((__local double *) scan_arr_mem_16668)[local_tid_16662] = x_13481;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_14145, n_13426)) {
            ((__global int32_t *) mem_16423)[gtid_14145] = ((__local
                                                             int32_t *) scan_arr_mem_16666)[local_tid_16662];
            ((__global double *) mem_16426)[gtid_14145] = ((__local
                                                            double *) scan_arr_mem_16668)[local_tid_16662];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_14141
}
__kernel void tridagParFlatziscan_stage2_14155(__global int *global_failure,
                                               __local volatile
                                               int64_t *scan_arr_mem_16798_backing_aligned_0,
                                               __local volatile
                                               int64_t *scan_arr_mem_16796_backing_aligned_1,
                                               __local volatile
                                               int64_t *scan_arr_mem_16794_backing_aligned_2,
                                               __local volatile
                                               int64_t *scan_arr_mem_16792_backing_aligned_3,
                                               __local volatile
                                               int64_t *scan_arr_mem_16790_backing_aligned_4,
                                               int32_t n_13426, __global
                                               unsigned char *mem_16430,
                                               __global
                                               unsigned char *mem_16433,
                                               __global
                                               unsigned char *mem_16436,
                                               __global
                                               unsigned char *mem_16439,
                                               __global
                                               unsigned char *mem_16442,
                                               int32_t stage1_num_groups_16695,
                                               int32_t num_threads_16696)
{
    #define segscan_group_sizze_14150 (tridagParFlatzisegscan_group_sizze_14149)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16798_backing_4 =
                          (__local volatile
                           char *) scan_arr_mem_16798_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16796_backing_3 =
                          (__local volatile
                           char *) scan_arr_mem_16796_backing_aligned_1;
    __local volatile char *restrict scan_arr_mem_16794_backing_2 =
                          (__local volatile
                           char *) scan_arr_mem_16794_backing_aligned_2;
    __local volatile char *restrict scan_arr_mem_16792_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16792_backing_aligned_3;
    __local volatile char *restrict scan_arr_mem_16790_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16790_backing_aligned_4;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16785;
    int32_t local_tid_16786;
    int32_t group_sizze_16789;
    int32_t wave_sizze_16788;
    int32_t group_tid_16787;
    
    global_tid_16785 = get_global_id(0);
    local_tid_16786 = get_local_id(0);
    group_sizze_16789 = get_local_size(0);
    wave_sizze_16788 = LOCKSTEP_WIDTH;
    group_tid_16787 = get_group_id(0);
    
    int32_t phys_tid_14155;
    
    phys_tid_14155 = global_tid_16785;
    
    __local char *scan_arr_mem_16790;
    __local char *scan_arr_mem_16792;
    __local char *scan_arr_mem_16794;
    __local char *scan_arr_mem_16796;
    __local char *scan_arr_mem_16798;
    
    scan_arr_mem_16790 = (__local char *) scan_arr_mem_16790_backing_0;
    scan_arr_mem_16792 = (__local char *) scan_arr_mem_16792_backing_1;
    scan_arr_mem_16794 = (__local char *) scan_arr_mem_16794_backing_2;
    scan_arr_mem_16796 = (__local char *) scan_arr_mem_16796_backing_3;
    scan_arr_mem_16798 = (__local char *) scan_arr_mem_16798_backing_4;
    
    int32_t flat_idx_16800;
    
    flat_idx_16800 = (local_tid_16786 + 1) * (segscan_group_sizze_14150 *
                                              sdiv_up32(n_13426,
                                                        num_threads_16696)) - 1;
    
    int32_t gtid_14154;
    
    gtid_14154 = flat_idx_16800;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_14154, n_13426)) {
            ((__local int32_t *) scan_arr_mem_16790)[local_tid_16786] =
                ((__global int32_t *) mem_16430)[gtid_14154];
            ((__local double *) scan_arr_mem_16792)[local_tid_16786] =
                ((__global double *) mem_16433)[gtid_14154];
            ((__local double *) scan_arr_mem_16794)[local_tid_16786] =
                ((__global double *) mem_16436)[gtid_14154];
            ((__local double *) scan_arr_mem_16796)[local_tid_16786] =
                ((__global double *) mem_16439)[gtid_14154];
            ((__local double *) scan_arr_mem_16798)[local_tid_16786] =
                ((__global double *) mem_16442)[gtid_14154];
        } else {
            ((__local int32_t *) scan_arr_mem_16790)[local_tid_16786] = 0;
            ((__local double *) scan_arr_mem_16792)[local_tid_16786] = 1.0;
            ((__local double *) scan_arr_mem_16794)[local_tid_16786] = 0.0;
            ((__local double *) scan_arr_mem_16796)[local_tid_16786] = 0.0;
            ((__local double *) scan_arr_mem_16798)[local_tid_16786] = 1.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t x_13504;
    double x_13505;
    double x_13506;
    double x_13507;
    double x_13508;
    int32_t x_13509;
    double x_13510;
    double x_13511;
    double x_13512;
    double x_13513;
    int32_t x_16801;
    double x_16802;
    double x_16803;
    double x_16804;
    double x_16805;
    int32_t x_16806;
    double x_16807;
    double x_16808;
    double x_16809;
    double x_16810;
    int32_t skip_threads_16834;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16786, stage1_num_groups_16695)) {
            x_13509 = ((volatile __local
                        int32_t *) scan_arr_mem_16790)[local_tid_16786];
            x_13510 = ((volatile __local
                        double *) scan_arr_mem_16792)[local_tid_16786];
            x_13511 = ((volatile __local
                        double *) scan_arr_mem_16794)[local_tid_16786];
            x_13512 = ((volatile __local
                        double *) scan_arr_mem_16796)[local_tid_16786];
            x_13513 = ((volatile __local
                        double *) scan_arr_mem_16798)[local_tid_16786];
            if ((local_tid_16786 - squot32(local_tid_16786, 32) * 32) == 0) {
                x_13504 = x_13509;
                x_13505 = x_13510;
                x_13506 = x_13511;
                x_13507 = x_13512;
                x_13508 = x_13513;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16834 = 1;
        while (slt32(skip_threads_16834, 32)) {
            if (sle32(skip_threads_16834, local_tid_16786 -
                      squot32(local_tid_16786, 32) * 32) &&
                slt32(local_tid_16786, stage1_num_groups_16695)) {
                // read operands
                {
                    x_13504 = ((volatile __local
                                int32_t *) scan_arr_mem_16790)[local_tid_16786 -
                                                               skip_threads_16834];
                    x_13505 = ((volatile __local
                                double *) scan_arr_mem_16792)[local_tid_16786 -
                                                              skip_threads_16834];
                    x_13506 = ((volatile __local
                                double *) scan_arr_mem_16794)[local_tid_16786 -
                                                              skip_threads_16834];
                    x_13507 = ((volatile __local
                                double *) scan_arr_mem_16796)[local_tid_16786 -
                                                              skip_threads_16834];
                    x_13508 = ((volatile __local
                                double *) scan_arr_mem_16798)[local_tid_16786 -
                                                              skip_threads_16834];
                }
                // perform operation
                {
                    int32_t f_13514 = x_13504 | x_13509;
                    bool cond_13515 = slt32(0, x_13509);
                    double res_13516;
                    double res_13517;
                    double res_13518;
                    double res_13519;
                    
                    if (cond_13515) {
                        res_13516 = x_13510;
                        res_13517 = x_13511;
                        res_13518 = x_13512;
                        res_13519 = x_13513;
                    } else {
                        double y_13520 = x_13505 * x_13510;
                        double value_13521 = 1.0 / y_13520;
                        double y_13522 = x_13507 * x_13511;
                        double x_13523 = y_13520 + y_13522;
                        double res_13524 = value_13521 * x_13523;
                        double x_13525 = x_13506 * x_13510;
                        double y_13526 = x_13508 * x_13511;
                        double x_13527 = x_13525 + y_13526;
                        double res_13528 = value_13521 * x_13527;
                        double x_13529 = x_13505 * x_13512;
                        double y_13530 = x_13507 * x_13513;
                        double x_13531 = x_13529 + y_13530;
                        double res_13532 = value_13521 * x_13531;
                        double x_13533 = x_13506 * x_13512;
                        double y_13534 = x_13508 * x_13513;
                        double x_13535 = x_13533 + y_13534;
                        double res_13536 = value_13521 * x_13535;
                        
                        res_13516 = res_13524;
                        res_13517 = res_13528;
                        res_13518 = res_13532;
                        res_13519 = res_13536;
                    }
                    x_13504 = f_13514;
                    x_13505 = res_13516;
                    x_13506 = res_13517;
                    x_13507 = res_13518;
                    x_13508 = res_13519;
                }
            }
            if (sle32(wave_sizze_16788, skip_threads_16834)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16834, local_tid_16786 -
                      squot32(local_tid_16786, 32) * 32) &&
                slt32(local_tid_16786, stage1_num_groups_16695)) {
                // write result
                {
                    ((volatile __local
                      int32_t *) scan_arr_mem_16790)[local_tid_16786] = x_13504;
                    x_13509 = x_13504;
                    ((volatile __local
                      double *) scan_arr_mem_16792)[local_tid_16786] = x_13505;
                    x_13510 = x_13505;
                    ((volatile __local
                      double *) scan_arr_mem_16794)[local_tid_16786] = x_13506;
                    x_13511 = x_13506;
                    ((volatile __local
                      double *) scan_arr_mem_16796)[local_tid_16786] = x_13507;
                    x_13512 = x_13507;
                    ((volatile __local
                      double *) scan_arr_mem_16798)[local_tid_16786] = x_13508;
                    x_13513 = x_13508;
                }
            }
            if (sle32(wave_sizze_16788, skip_threads_16834)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16834 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16786 - squot32(local_tid_16786, 32) * 32) == 31 &&
            slt32(local_tid_16786, stage1_num_groups_16695)) {
            ((volatile __local
              int32_t *) scan_arr_mem_16790)[squot32(local_tid_16786, 32)] =
                x_13504;
            ((volatile __local
              double *) scan_arr_mem_16792)[squot32(local_tid_16786, 32)] =
                x_13505;
            ((volatile __local
              double *) scan_arr_mem_16794)[squot32(local_tid_16786, 32)] =
                x_13506;
            ((volatile __local
              double *) scan_arr_mem_16796)[squot32(local_tid_16786, 32)] =
                x_13507;
            ((volatile __local
              double *) scan_arr_mem_16798)[squot32(local_tid_16786, 32)] =
                x_13508;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16835;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16786, 32) == 0 && slt32(local_tid_16786,
                                                           stage1_num_groups_16695)) {
                x_16806 = ((volatile __local
                            int32_t *) scan_arr_mem_16790)[local_tid_16786];
                x_16807 = ((volatile __local
                            double *) scan_arr_mem_16792)[local_tid_16786];
                x_16808 = ((volatile __local
                            double *) scan_arr_mem_16794)[local_tid_16786];
                x_16809 = ((volatile __local
                            double *) scan_arr_mem_16796)[local_tid_16786];
                x_16810 = ((volatile __local
                            double *) scan_arr_mem_16798)[local_tid_16786];
                if ((local_tid_16786 - squot32(local_tid_16786, 32) * 32) ==
                    0) {
                    x_16801 = x_16806;
                    x_16802 = x_16807;
                    x_16803 = x_16808;
                    x_16804 = x_16809;
                    x_16805 = x_16810;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16835 = 1;
            while (slt32(skip_threads_16835, 32)) {
                if (sle32(skip_threads_16835, local_tid_16786 -
                          squot32(local_tid_16786, 32) * 32) &&
                    (squot32(local_tid_16786, 32) == 0 && slt32(local_tid_16786,
                                                                stage1_num_groups_16695))) {
                    // read operands
                    {
                        x_16801 = ((volatile __local
                                    int32_t *) scan_arr_mem_16790)[local_tid_16786 -
                                                                   skip_threads_16835];
                        x_16802 = ((volatile __local
                                    double *) scan_arr_mem_16792)[local_tid_16786 -
                                                                  skip_threads_16835];
                        x_16803 = ((volatile __local
                                    double *) scan_arr_mem_16794)[local_tid_16786 -
                                                                  skip_threads_16835];
                        x_16804 = ((volatile __local
                                    double *) scan_arr_mem_16796)[local_tid_16786 -
                                                                  skip_threads_16835];
                        x_16805 = ((volatile __local
                                    double *) scan_arr_mem_16798)[local_tid_16786 -
                                                                  skip_threads_16835];
                    }
                    // perform operation
                    {
                        int32_t f_16811 = x_16801 | x_16806;
                        bool cond_16812 = slt32(0, x_16806);
                        double res_16813;
                        double res_16814;
                        double res_16815;
                        double res_16816;
                        
                        if (cond_16812) {
                            res_16813 = x_16807;
                            res_16814 = x_16808;
                            res_16815 = x_16809;
                            res_16816 = x_16810;
                        } else {
                            double y_16817 = x_16802 * x_16807;
                            double value_16818 = 1.0 / y_16817;
                            double y_16819 = x_16804 * x_16808;
                            double x_16820 = y_16817 + y_16819;
                            double res_16821 = value_16818 * x_16820;
                            double x_16822 = x_16803 * x_16807;
                            double y_16823 = x_16805 * x_16808;
                            double x_16824 = x_16822 + y_16823;
                            double res_16825 = value_16818 * x_16824;
                            double x_16826 = x_16802 * x_16809;
                            double y_16827 = x_16804 * x_16810;
                            double x_16828 = x_16826 + y_16827;
                            double res_16829 = value_16818 * x_16828;
                            double x_16830 = x_16803 * x_16809;
                            double y_16831 = x_16805 * x_16810;
                            double x_16832 = x_16830 + y_16831;
                            double res_16833 = value_16818 * x_16832;
                            
                            res_16813 = res_16821;
                            res_16814 = res_16825;
                            res_16815 = res_16829;
                            res_16816 = res_16833;
                        }
                        x_16801 = f_16811;
                        x_16802 = res_16813;
                        x_16803 = res_16814;
                        x_16804 = res_16815;
                        x_16805 = res_16816;
                    }
                }
                if (sle32(wave_sizze_16788, skip_threads_16835)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16835, local_tid_16786 -
                          squot32(local_tid_16786, 32) * 32) &&
                    (squot32(local_tid_16786, 32) == 0 && slt32(local_tid_16786,
                                                                stage1_num_groups_16695))) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) scan_arr_mem_16790)[local_tid_16786] =
                            x_16801;
                        x_16806 = x_16801;
                        ((volatile __local
                          double *) scan_arr_mem_16792)[local_tid_16786] =
                            x_16802;
                        x_16807 = x_16802;
                        ((volatile __local
                          double *) scan_arr_mem_16794)[local_tid_16786] =
                            x_16803;
                        x_16808 = x_16803;
                        ((volatile __local
                          double *) scan_arr_mem_16796)[local_tid_16786] =
                            x_16804;
                        x_16809 = x_16804;
                        ((volatile __local
                          double *) scan_arr_mem_16798)[local_tid_16786] =
                            x_16805;
                        x_16810 = x_16805;
                    }
                }
                if (sle32(wave_sizze_16788, skip_threads_16835)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16835 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16786, 32) == 0 || !slt32(local_tid_16786,
                                                          stage1_num_groups_16695))) {
            // read operands
            {
                x_13509 = x_13504;
                x_13510 = x_13505;
                x_13511 = x_13506;
                x_13512 = x_13507;
                x_13513 = x_13508;
                x_13504 = ((__local
                            int32_t *) scan_arr_mem_16790)[squot32(local_tid_16786,
                                                                   32) - 1];
                x_13505 = ((__local
                            double *) scan_arr_mem_16792)[squot32(local_tid_16786,
                                                                  32) - 1];
                x_13506 = ((__local
                            double *) scan_arr_mem_16794)[squot32(local_tid_16786,
                                                                  32) - 1];
                x_13507 = ((__local
                            double *) scan_arr_mem_16796)[squot32(local_tid_16786,
                                                                  32) - 1];
                x_13508 = ((__local
                            double *) scan_arr_mem_16798)[squot32(local_tid_16786,
                                                                  32) - 1];
            }
            // perform operation
            {
                int32_t f_13514 = x_13504 | x_13509;
                bool cond_13515 = slt32(0, x_13509);
                double res_13516;
                double res_13517;
                double res_13518;
                double res_13519;
                
                if (cond_13515) {
                    res_13516 = x_13510;
                    res_13517 = x_13511;
                    res_13518 = x_13512;
                    res_13519 = x_13513;
                } else {
                    double y_13520 = x_13505 * x_13510;
                    double value_13521 = 1.0 / y_13520;
                    double y_13522 = x_13507 * x_13511;
                    double x_13523 = y_13520 + y_13522;
                    double res_13524 = value_13521 * x_13523;
                    double x_13525 = x_13506 * x_13510;
                    double y_13526 = x_13508 * x_13511;
                    double x_13527 = x_13525 + y_13526;
                    double res_13528 = value_13521 * x_13527;
                    double x_13529 = x_13505 * x_13512;
                    double y_13530 = x_13507 * x_13513;
                    double x_13531 = x_13529 + y_13530;
                    double res_13532 = value_13521 * x_13531;
                    double x_13533 = x_13506 * x_13512;
                    double y_13534 = x_13508 * x_13513;
                    double x_13535 = x_13533 + y_13534;
                    double res_13536 = value_13521 * x_13535;
                    
                    res_13516 = res_13524;
                    res_13517 = res_13528;
                    res_13518 = res_13532;
                    res_13519 = res_13536;
                }
                x_13504 = f_13514;
                x_13505 = res_13516;
                x_13506 = res_13517;
                x_13507 = res_13518;
                x_13508 = res_13519;
            }
            // write final result
            {
                ((__local int32_t *) scan_arr_mem_16790)[local_tid_16786] =
                    x_13504;
                ((__local double *) scan_arr_mem_16792)[local_tid_16786] =
                    x_13505;
                ((__local double *) scan_arr_mem_16794)[local_tid_16786] =
                    x_13506;
                ((__local double *) scan_arr_mem_16796)[local_tid_16786] =
                    x_13507;
                ((__local double *) scan_arr_mem_16798)[local_tid_16786] =
                    x_13508;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16786, 32) == 0) {
            ((__local int32_t *) scan_arr_mem_16790)[local_tid_16786] = x_13509;
            ((__local double *) scan_arr_mem_16792)[local_tid_16786] = x_13510;
            ((__local double *) scan_arr_mem_16794)[local_tid_16786] = x_13511;
            ((__local double *) scan_arr_mem_16796)[local_tid_16786] = x_13512;
            ((__local double *) scan_arr_mem_16798)[local_tid_16786] = x_13513;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_14154, n_13426)) {
            ((__global int32_t *) mem_16430)[gtid_14154] = ((__local
                                                             int32_t *) scan_arr_mem_16790)[local_tid_16786];
            ((__global double *) mem_16433)[gtid_14154] = ((__local
                                                            double *) scan_arr_mem_16792)[local_tid_16786];
            ((__global double *) mem_16436)[gtid_14154] = ((__local
                                                            double *) scan_arr_mem_16794)[local_tid_16786];
            ((__global double *) mem_16439)[gtid_14154] = ((__local
                                                            double *) scan_arr_mem_16796)[local_tid_16786];
            ((__global double *) mem_16442)[gtid_14154] = ((__local
                                                            double *) scan_arr_mem_16798)[local_tid_16786];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_14150
}
__kernel void tridagParFlatziscan_stage2_14355(__global int *global_failure,
                                               __local volatile
                                               int64_t *scan_arr_mem_16901_backing_aligned_0,
                                               __local volatile
                                               int64_t *scan_arr_mem_16899_backing_aligned_1,
                                               int32_t n_13426, __global
                                               unsigned char *mem_16454,
                                               __global
                                               unsigned char *mem_16457,
                                               int32_t stage1_num_groups_16860,
                                               int32_t num_threads_16861)
{
    #define segscan_group_sizze_14350 (tridagParFlatzisegscan_group_sizze_14349)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16901_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16901_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16899_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16899_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16894;
    int32_t local_tid_16895;
    int32_t group_sizze_16898;
    int32_t wave_sizze_16897;
    int32_t group_tid_16896;
    
    global_tid_16894 = get_global_id(0);
    local_tid_16895 = get_local_id(0);
    group_sizze_16898 = get_local_size(0);
    wave_sizze_16897 = LOCKSTEP_WIDTH;
    group_tid_16896 = get_group_id(0);
    
    int32_t phys_tid_14355;
    
    phys_tid_14355 = global_tid_16894;
    
    __local char *scan_arr_mem_16899;
    __local char *scan_arr_mem_16901;
    
    scan_arr_mem_16899 = (__local char *) scan_arr_mem_16899_backing_0;
    scan_arr_mem_16901 = (__local char *) scan_arr_mem_16901_backing_1;
    
    int32_t flat_idx_16903;
    
    flat_idx_16903 = (local_tid_16895 + 1) * (segscan_group_sizze_14350 *
                                              sdiv_up32(n_13426,
                                                        num_threads_16861)) - 1;
    
    int32_t gtid_14354;
    
    gtid_14354 = flat_idx_16903;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_14354, n_13426)) {
            ((__local int32_t *) scan_arr_mem_16899)[local_tid_16895] =
                ((__global int32_t *) mem_16454)[gtid_14354];
            ((__local double *) scan_arr_mem_16901)[local_tid_16895] =
                ((__global double *) mem_16457)[gtid_14354];
        } else {
            ((__local int32_t *) scan_arr_mem_16899)[local_tid_16895] = 0;
            ((__local double *) scan_arr_mem_16901)[local_tid_16895] = 0.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t x_13646;
    double x_13647;
    int32_t x_13648;
    double x_13649;
    int32_t x_16904;
    double x_16905;
    int32_t x_16906;
    double x_16907;
    int32_t skip_threads_16912;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16895, stage1_num_groups_16860)) {
            x_13648 = ((volatile __local
                        int32_t *) scan_arr_mem_16899)[local_tid_16895];
            x_13649 = ((volatile __local
                        double *) scan_arr_mem_16901)[local_tid_16895];
            if ((local_tid_16895 - squot32(local_tid_16895, 32) * 32) == 0) {
                x_13646 = x_13648;
                x_13647 = x_13649;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16912 = 1;
        while (slt32(skip_threads_16912, 32)) {
            if (sle32(skip_threads_16912, local_tid_16895 -
                      squot32(local_tid_16895, 32) * 32) &&
                slt32(local_tid_16895, stage1_num_groups_16860)) {
                // read operands
                {
                    x_13646 = ((volatile __local
                                int32_t *) scan_arr_mem_16899)[local_tid_16895 -
                                                               skip_threads_16912];
                    x_13647 = ((volatile __local
                                double *) scan_arr_mem_16901)[local_tid_16895 -
                                                              skip_threads_16912];
                }
                // perform operation
                {
                    int32_t f_13650 = x_13646 | x_13648;
                    bool cond_13651 = slt32(0, x_13648);
                    double res_13652;
                    
                    if (cond_13651) {
                        res_13652 = x_13649;
                    } else {
                        double res_13653 = x_13647 + x_13649;
                        
                        res_13652 = res_13653;
                    }
                    x_13646 = f_13650;
                    x_13647 = res_13652;
                }
            }
            if (sle32(wave_sizze_16897, skip_threads_16912)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16912, local_tid_16895 -
                      squot32(local_tid_16895, 32) * 32) &&
                slt32(local_tid_16895, stage1_num_groups_16860)) {
                // write result
                {
                    ((volatile __local
                      int32_t *) scan_arr_mem_16899)[local_tid_16895] = x_13646;
                    x_13648 = x_13646;
                    ((volatile __local
                      double *) scan_arr_mem_16901)[local_tid_16895] = x_13647;
                    x_13649 = x_13647;
                }
            }
            if (sle32(wave_sizze_16897, skip_threads_16912)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16912 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16895 - squot32(local_tid_16895, 32) * 32) == 31 &&
            slt32(local_tid_16895, stage1_num_groups_16860)) {
            ((volatile __local
              int32_t *) scan_arr_mem_16899)[squot32(local_tid_16895, 32)] =
                x_13646;
            ((volatile __local
              double *) scan_arr_mem_16901)[squot32(local_tid_16895, 32)] =
                x_13647;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16913;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16895, 32) == 0 && slt32(local_tid_16895,
                                                           stage1_num_groups_16860)) {
                x_16906 = ((volatile __local
                            int32_t *) scan_arr_mem_16899)[local_tid_16895];
                x_16907 = ((volatile __local
                            double *) scan_arr_mem_16901)[local_tid_16895];
                if ((local_tid_16895 - squot32(local_tid_16895, 32) * 32) ==
                    0) {
                    x_16904 = x_16906;
                    x_16905 = x_16907;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16913 = 1;
            while (slt32(skip_threads_16913, 32)) {
                if (sle32(skip_threads_16913, local_tid_16895 -
                          squot32(local_tid_16895, 32) * 32) &&
                    (squot32(local_tid_16895, 32) == 0 && slt32(local_tid_16895,
                                                                stage1_num_groups_16860))) {
                    // read operands
                    {
                        x_16904 = ((volatile __local
                                    int32_t *) scan_arr_mem_16899)[local_tid_16895 -
                                                                   skip_threads_16913];
                        x_16905 = ((volatile __local
                                    double *) scan_arr_mem_16901)[local_tid_16895 -
                                                                  skip_threads_16913];
                    }
                    // perform operation
                    {
                        int32_t f_16908 = x_16904 | x_16906;
                        bool cond_16909 = slt32(0, x_16906);
                        double res_16910;
                        
                        if (cond_16909) {
                            res_16910 = x_16907;
                        } else {
                            double res_16911 = x_16905 + x_16907;
                            
                            res_16910 = res_16911;
                        }
                        x_16904 = f_16908;
                        x_16905 = res_16910;
                    }
                }
                if (sle32(wave_sizze_16897, skip_threads_16913)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16913, local_tid_16895 -
                          squot32(local_tid_16895, 32) * 32) &&
                    (squot32(local_tid_16895, 32) == 0 && slt32(local_tid_16895,
                                                                stage1_num_groups_16860))) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) scan_arr_mem_16899)[local_tid_16895] =
                            x_16904;
                        x_16906 = x_16904;
                        ((volatile __local
                          double *) scan_arr_mem_16901)[local_tid_16895] =
                            x_16905;
                        x_16907 = x_16905;
                    }
                }
                if (sle32(wave_sizze_16897, skip_threads_16913)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16913 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16895, 32) == 0 || !slt32(local_tid_16895,
                                                          stage1_num_groups_16860))) {
            // read operands
            {
                x_13648 = x_13646;
                x_13649 = x_13647;
                x_13646 = ((__local
                            int32_t *) scan_arr_mem_16899)[squot32(local_tid_16895,
                                                                   32) - 1];
                x_13647 = ((__local
                            double *) scan_arr_mem_16901)[squot32(local_tid_16895,
                                                                  32) - 1];
            }
            // perform operation
            {
                int32_t f_13650 = x_13646 | x_13648;
                bool cond_13651 = slt32(0, x_13648);
                double res_13652;
                
                if (cond_13651) {
                    res_13652 = x_13649;
                } else {
                    double res_13653 = x_13647 + x_13649;
                    
                    res_13652 = res_13653;
                }
                x_13646 = f_13650;
                x_13647 = res_13652;
            }
            // write final result
            {
                ((__local int32_t *) scan_arr_mem_16899)[local_tid_16895] =
                    x_13646;
                ((__local double *) scan_arr_mem_16901)[local_tid_16895] =
                    x_13647;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16895, 32) == 0) {
            ((__local int32_t *) scan_arr_mem_16899)[local_tid_16895] = x_13648;
            ((__local double *) scan_arr_mem_16901)[local_tid_16895] = x_13649;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_14354, n_13426)) {
            ((__global int32_t *) mem_16454)[gtid_14354] = ((__local
                                                             int32_t *) scan_arr_mem_16899)[local_tid_16895];
            ((__global double *) mem_16457)[gtid_14354] = ((__local
                                                            double *) scan_arr_mem_16901)[local_tid_16895];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_14350
}
__kernel void tridagParFlatziscan_stage2_14364(__global int *global_failure,
                                               __local volatile
                                               int64_t *scan_arr_mem_16983_backing_aligned_0,
                                               __local volatile
                                               int64_t *scan_arr_mem_16981_backing_aligned_1,
                                               __local volatile
                                               int64_t *scan_arr_mem_16979_backing_aligned_2,
                                               int32_t n_13426, __global
                                               unsigned char *mem_16461,
                                               __global
                                               unsigned char *mem_16464,
                                               __global
                                               unsigned char *mem_16467,
                                               int32_t stage1_num_groups_16928,
                                               int32_t num_threads_16929)
{
    #define segscan_group_sizze_14359 (tridagParFlatzisegscan_group_sizze_14358)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16983_backing_2 =
                          (__local volatile
                           char *) scan_arr_mem_16983_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16981_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16981_backing_aligned_1;
    __local volatile char *restrict scan_arr_mem_16979_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16979_backing_aligned_2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16974;
    int32_t local_tid_16975;
    int32_t group_sizze_16978;
    int32_t wave_sizze_16977;
    int32_t group_tid_16976;
    
    global_tid_16974 = get_global_id(0);
    local_tid_16975 = get_local_id(0);
    group_sizze_16978 = get_local_size(0);
    wave_sizze_16977 = LOCKSTEP_WIDTH;
    group_tid_16976 = get_group_id(0);
    
    int32_t phys_tid_14364;
    
    phys_tid_14364 = global_tid_16974;
    
    __local char *scan_arr_mem_16979;
    __local char *scan_arr_mem_16981;
    __local char *scan_arr_mem_16983;
    
    scan_arr_mem_16979 = (__local char *) scan_arr_mem_16979_backing_0;
    scan_arr_mem_16981 = (__local char *) scan_arr_mem_16981_backing_1;
    scan_arr_mem_16983 = (__local char *) scan_arr_mem_16983_backing_2;
    
    int32_t flat_idx_16985;
    
    flat_idx_16985 = (local_tid_16975 + 1) * (segscan_group_sizze_14359 *
                                              sdiv_up32(n_13426,
                                                        num_threads_16929)) - 1;
    
    int32_t gtid_14363;
    
    gtid_14363 = flat_idx_16985;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_14363, n_13426)) {
            ((__local int32_t *) scan_arr_mem_16979)[local_tid_16975] =
                ((__global int32_t *) mem_16461)[gtid_14363];
            ((__local double *) scan_arr_mem_16981)[local_tid_16975] =
                ((__global double *) mem_16464)[gtid_14363];
            ((__local double *) scan_arr_mem_16983)[local_tid_16975] =
                ((__global double *) mem_16467)[gtid_14363];
        } else {
            ((__local int32_t *) scan_arr_mem_16979)[local_tid_16975] = 0;
            ((__local double *) scan_arr_mem_16981)[local_tid_16975] = 0.0;
            ((__local double *) scan_arr_mem_16983)[local_tid_16975] = 1.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t x_13670;
    double x_13671;
    double x_13672;
    int32_t x_13673;
    double x_13674;
    double x_13675;
    int32_t x_16986;
    double x_16987;
    double x_16988;
    int32_t x_16989;
    double x_16990;
    double x_16991;
    int32_t skip_threads_16999;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16975, stage1_num_groups_16928)) {
            x_13673 = ((volatile __local
                        int32_t *) scan_arr_mem_16979)[local_tid_16975];
            x_13674 = ((volatile __local
                        double *) scan_arr_mem_16981)[local_tid_16975];
            x_13675 = ((volatile __local
                        double *) scan_arr_mem_16983)[local_tid_16975];
            if ((local_tid_16975 - squot32(local_tid_16975, 32) * 32) == 0) {
                x_13670 = x_13673;
                x_13671 = x_13674;
                x_13672 = x_13675;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16999 = 1;
        while (slt32(skip_threads_16999, 32)) {
            if (sle32(skip_threads_16999, local_tid_16975 -
                      squot32(local_tid_16975, 32) * 32) &&
                slt32(local_tid_16975, stage1_num_groups_16928)) {
                // read operands
                {
                    x_13670 = ((volatile __local
                                int32_t *) scan_arr_mem_16979)[local_tid_16975 -
                                                               skip_threads_16999];
                    x_13671 = ((volatile __local
                                double *) scan_arr_mem_16981)[local_tid_16975 -
                                                              skip_threads_16999];
                    x_13672 = ((volatile __local
                                double *) scan_arr_mem_16983)[local_tid_16975 -
                                                              skip_threads_16999];
                }
                // perform operation
                {
                    int32_t f_13676 = x_13670 | x_13673;
                    bool cond_13677 = slt32(0, x_13673);
                    double res_13678;
                    double res_13679;
                    
                    if (cond_13677) {
                        res_13678 = x_13674;
                        res_13679 = x_13675;
                    } else {
                        double y_13680 = x_13671 * x_13675;
                        double res_13681 = x_13674 + y_13680;
                        double res_13682 = x_13672 * x_13675;
                        
                        res_13678 = res_13681;
                        res_13679 = res_13682;
                    }
                    x_13670 = f_13676;
                    x_13671 = res_13678;
                    x_13672 = res_13679;
                }
            }
            if (sle32(wave_sizze_16977, skip_threads_16999)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16999, local_tid_16975 -
                      squot32(local_tid_16975, 32) * 32) &&
                slt32(local_tid_16975, stage1_num_groups_16928)) {
                // write result
                {
                    ((volatile __local
                      int32_t *) scan_arr_mem_16979)[local_tid_16975] = x_13670;
                    x_13673 = x_13670;
                    ((volatile __local
                      double *) scan_arr_mem_16981)[local_tid_16975] = x_13671;
                    x_13674 = x_13671;
                    ((volatile __local
                      double *) scan_arr_mem_16983)[local_tid_16975] = x_13672;
                    x_13675 = x_13672;
                }
            }
            if (sle32(wave_sizze_16977, skip_threads_16999)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16999 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16975 - squot32(local_tid_16975, 32) * 32) == 31 &&
            slt32(local_tid_16975, stage1_num_groups_16928)) {
            ((volatile __local
              int32_t *) scan_arr_mem_16979)[squot32(local_tid_16975, 32)] =
                x_13670;
            ((volatile __local
              double *) scan_arr_mem_16981)[squot32(local_tid_16975, 32)] =
                x_13671;
            ((volatile __local
              double *) scan_arr_mem_16983)[squot32(local_tid_16975, 32)] =
                x_13672;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_17000;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16975, 32) == 0 && slt32(local_tid_16975,
                                                           stage1_num_groups_16928)) {
                x_16989 = ((volatile __local
                            int32_t *) scan_arr_mem_16979)[local_tid_16975];
                x_16990 = ((volatile __local
                            double *) scan_arr_mem_16981)[local_tid_16975];
                x_16991 = ((volatile __local
                            double *) scan_arr_mem_16983)[local_tid_16975];
                if ((local_tid_16975 - squot32(local_tid_16975, 32) * 32) ==
                    0) {
                    x_16986 = x_16989;
                    x_16987 = x_16990;
                    x_16988 = x_16991;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_17000 = 1;
            while (slt32(skip_threads_17000, 32)) {
                if (sle32(skip_threads_17000, local_tid_16975 -
                          squot32(local_tid_16975, 32) * 32) &&
                    (squot32(local_tid_16975, 32) == 0 && slt32(local_tid_16975,
                                                                stage1_num_groups_16928))) {
                    // read operands
                    {
                        x_16986 = ((volatile __local
                                    int32_t *) scan_arr_mem_16979)[local_tid_16975 -
                                                                   skip_threads_17000];
                        x_16987 = ((volatile __local
                                    double *) scan_arr_mem_16981)[local_tid_16975 -
                                                                  skip_threads_17000];
                        x_16988 = ((volatile __local
                                    double *) scan_arr_mem_16983)[local_tid_16975 -
                                                                  skip_threads_17000];
                    }
                    // perform operation
                    {
                        int32_t f_16992 = x_16986 | x_16989;
                        bool cond_16993 = slt32(0, x_16989);
                        double res_16994;
                        double res_16995;
                        
                        if (cond_16993) {
                            res_16994 = x_16990;
                            res_16995 = x_16991;
                        } else {
                            double y_16996 = x_16987 * x_16991;
                            double res_16997 = x_16990 + y_16996;
                            double res_16998 = x_16988 * x_16991;
                            
                            res_16994 = res_16997;
                            res_16995 = res_16998;
                        }
                        x_16986 = f_16992;
                        x_16987 = res_16994;
                        x_16988 = res_16995;
                    }
                }
                if (sle32(wave_sizze_16977, skip_threads_17000)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_17000, local_tid_16975 -
                          squot32(local_tid_16975, 32) * 32) &&
                    (squot32(local_tid_16975, 32) == 0 && slt32(local_tid_16975,
                                                                stage1_num_groups_16928))) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) scan_arr_mem_16979)[local_tid_16975] =
                            x_16986;
                        x_16989 = x_16986;
                        ((volatile __local
                          double *) scan_arr_mem_16981)[local_tid_16975] =
                            x_16987;
                        x_16990 = x_16987;
                        ((volatile __local
                          double *) scan_arr_mem_16983)[local_tid_16975] =
                            x_16988;
                        x_16991 = x_16988;
                    }
                }
                if (sle32(wave_sizze_16977, skip_threads_17000)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_17000 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16975, 32) == 0 || !slt32(local_tid_16975,
                                                          stage1_num_groups_16928))) {
            // read operands
            {
                x_13673 = x_13670;
                x_13674 = x_13671;
                x_13675 = x_13672;
                x_13670 = ((__local
                            int32_t *) scan_arr_mem_16979)[squot32(local_tid_16975,
                                                                   32) - 1];
                x_13671 = ((__local
                            double *) scan_arr_mem_16981)[squot32(local_tid_16975,
                                                                  32) - 1];
                x_13672 = ((__local
                            double *) scan_arr_mem_16983)[squot32(local_tid_16975,
                                                                  32) - 1];
            }
            // perform operation
            {
                int32_t f_13676 = x_13670 | x_13673;
                bool cond_13677 = slt32(0, x_13673);
                double res_13678;
                double res_13679;
                
                if (cond_13677) {
                    res_13678 = x_13674;
                    res_13679 = x_13675;
                } else {
                    double y_13680 = x_13671 * x_13675;
                    double res_13681 = x_13674 + y_13680;
                    double res_13682 = x_13672 * x_13675;
                    
                    res_13678 = res_13681;
                    res_13679 = res_13682;
                }
                x_13670 = f_13676;
                x_13671 = res_13678;
                x_13672 = res_13679;
            }
            // write final result
            {
                ((__local int32_t *) scan_arr_mem_16979)[local_tid_16975] =
                    x_13670;
                ((__local double *) scan_arr_mem_16981)[local_tid_16975] =
                    x_13671;
                ((__local double *) scan_arr_mem_16983)[local_tid_16975] =
                    x_13672;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16975, 32) == 0) {
            ((__local int32_t *) scan_arr_mem_16979)[local_tid_16975] = x_13673;
            ((__local double *) scan_arr_mem_16981)[local_tid_16975] = x_13674;
            ((__local double *) scan_arr_mem_16983)[local_tid_16975] = x_13675;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_14363, n_13426)) {
            ((__global int32_t *) mem_16461)[gtid_14363] = ((__local
                                                             int32_t *) scan_arr_mem_16979)[local_tid_16975];
            ((__global double *) mem_16464)[gtid_14363] = ((__local
                                                            double *) scan_arr_mem_16981)[local_tid_16975];
            ((__global double *) mem_16467)[gtid_14363] = ((__local
                                                            double *) scan_arr_mem_16983)[local_tid_16975];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_14359
}
__kernel void tridagParFlatziscan_stage2_14480(__global int *global_failure,
                                               __local volatile
                                               int64_t *scan_arr_mem_17066_backing_aligned_0,
                                               __local volatile
                                               int64_t *scan_arr_mem_17064_backing_aligned_1,
                                               int32_t n_13426, __global
                                               unsigned char *mem_16479,
                                               __global
                                               unsigned char *mem_16482,
                                               int32_t stage1_num_groups_17025,
                                               int32_t num_threads_17026)
{
    #define segscan_group_sizze_14475 (tridagParFlatzisegscan_group_sizze_14474)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17066_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17066_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17064_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17064_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17059;
    int32_t local_tid_17060;
    int32_t group_sizze_17063;
    int32_t wave_sizze_17062;
    int32_t group_tid_17061;
    
    global_tid_17059 = get_global_id(0);
    local_tid_17060 = get_local_id(0);
    group_sizze_17063 = get_local_size(0);
    wave_sizze_17062 = LOCKSTEP_WIDTH;
    group_tid_17061 = get_group_id(0);
    
    int32_t phys_tid_14480;
    
    phys_tid_14480 = global_tid_17059;
    
    __local char *scan_arr_mem_17064;
    __local char *scan_arr_mem_17066;
    
    scan_arr_mem_17064 = (__local char *) scan_arr_mem_17064_backing_0;
    scan_arr_mem_17066 = (__local char *) scan_arr_mem_17066_backing_1;
    
    int32_t flat_idx_17068;
    
    flat_idx_17068 = (local_tid_17060 + 1) * (segscan_group_sizze_14475 *
                                              sdiv_up32(n_13426,
                                                        num_threads_17026)) - 1;
    
    int32_t gtid_14479;
    
    gtid_14479 = flat_idx_17068;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_14479, n_13426)) {
            ((__local int32_t *) scan_arr_mem_17064)[local_tid_17060] =
                ((__global int32_t *) mem_16479)[gtid_14479];
            ((__local double *) scan_arr_mem_17066)[local_tid_17060] =
                ((__global double *) mem_16482)[gtid_14479];
        } else {
            ((__local int32_t *) scan_arr_mem_17064)[local_tid_17060] = 0;
            ((__local double *) scan_arr_mem_17066)[local_tid_17060] = 0.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t x_13750;
    double x_13751;
    int32_t x_13752;
    double x_13753;
    int32_t x_17069;
    double x_17070;
    int32_t x_17071;
    double x_17072;
    int32_t skip_threads_17077;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_17060, stage1_num_groups_17025)) {
            x_13752 = ((volatile __local
                        int32_t *) scan_arr_mem_17064)[local_tid_17060];
            x_13753 = ((volatile __local
                        double *) scan_arr_mem_17066)[local_tid_17060];
            if ((local_tid_17060 - squot32(local_tid_17060, 32) * 32) == 0) {
                x_13750 = x_13752;
                x_13751 = x_13753;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_17077 = 1;
        while (slt32(skip_threads_17077, 32)) {
            if (sle32(skip_threads_17077, local_tid_17060 -
                      squot32(local_tid_17060, 32) * 32) &&
                slt32(local_tid_17060, stage1_num_groups_17025)) {
                // read operands
                {
                    x_13750 = ((volatile __local
                                int32_t *) scan_arr_mem_17064)[local_tid_17060 -
                                                               skip_threads_17077];
                    x_13751 = ((volatile __local
                                double *) scan_arr_mem_17066)[local_tid_17060 -
                                                              skip_threads_17077];
                }
                // perform operation
                {
                    int32_t f_13754 = x_13750 | x_13752;
                    bool cond_13755 = slt32(0, x_13752);
                    double res_13756;
                    
                    if (cond_13755) {
                        res_13756 = x_13753;
                    } else {
                        double res_13757 = x_13751 + x_13753;
                        
                        res_13756 = res_13757;
                    }
                    x_13750 = f_13754;
                    x_13751 = res_13756;
                }
            }
            if (sle32(wave_sizze_17062, skip_threads_17077)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_17077, local_tid_17060 -
                      squot32(local_tid_17060, 32) * 32) &&
                slt32(local_tid_17060, stage1_num_groups_17025)) {
                // write result
                {
                    ((volatile __local
                      int32_t *) scan_arr_mem_17064)[local_tid_17060] = x_13750;
                    x_13752 = x_13750;
                    ((volatile __local
                      double *) scan_arr_mem_17066)[local_tid_17060] = x_13751;
                    x_13753 = x_13751;
                }
            }
            if (sle32(wave_sizze_17062, skip_threads_17077)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_17077 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_17060 - squot32(local_tid_17060, 32) * 32) == 31 &&
            slt32(local_tid_17060, stage1_num_groups_17025)) {
            ((volatile __local
              int32_t *) scan_arr_mem_17064)[squot32(local_tid_17060, 32)] =
                x_13750;
            ((volatile __local
              double *) scan_arr_mem_17066)[squot32(local_tid_17060, 32)] =
                x_13751;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_17078;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_17060, 32) == 0 && slt32(local_tid_17060,
                                                           stage1_num_groups_17025)) {
                x_17071 = ((volatile __local
                            int32_t *) scan_arr_mem_17064)[local_tid_17060];
                x_17072 = ((volatile __local
                            double *) scan_arr_mem_17066)[local_tid_17060];
                if ((local_tid_17060 - squot32(local_tid_17060, 32) * 32) ==
                    0) {
                    x_17069 = x_17071;
                    x_17070 = x_17072;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_17078 = 1;
            while (slt32(skip_threads_17078, 32)) {
                if (sle32(skip_threads_17078, local_tid_17060 -
                          squot32(local_tid_17060, 32) * 32) &&
                    (squot32(local_tid_17060, 32) == 0 && slt32(local_tid_17060,
                                                                stage1_num_groups_17025))) {
                    // read operands
                    {
                        x_17069 = ((volatile __local
                                    int32_t *) scan_arr_mem_17064)[local_tid_17060 -
                                                                   skip_threads_17078];
                        x_17070 = ((volatile __local
                                    double *) scan_arr_mem_17066)[local_tid_17060 -
                                                                  skip_threads_17078];
                    }
                    // perform operation
                    {
                        int32_t f_17073 = x_17069 | x_17071;
                        bool cond_17074 = slt32(0, x_17071);
                        double res_17075;
                        
                        if (cond_17074) {
                            res_17075 = x_17072;
                        } else {
                            double res_17076 = x_17070 + x_17072;
                            
                            res_17075 = res_17076;
                        }
                        x_17069 = f_17073;
                        x_17070 = res_17075;
                    }
                }
                if (sle32(wave_sizze_17062, skip_threads_17078)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_17078, local_tid_17060 -
                          squot32(local_tid_17060, 32) * 32) &&
                    (squot32(local_tid_17060, 32) == 0 && slt32(local_tid_17060,
                                                                stage1_num_groups_17025))) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) scan_arr_mem_17064)[local_tid_17060] =
                            x_17069;
                        x_17071 = x_17069;
                        ((volatile __local
                          double *) scan_arr_mem_17066)[local_tid_17060] =
                            x_17070;
                        x_17072 = x_17070;
                    }
                }
                if (sle32(wave_sizze_17062, skip_threads_17078)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_17078 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_17060, 32) == 0 || !slt32(local_tid_17060,
                                                          stage1_num_groups_17025))) {
            // read operands
            {
                x_13752 = x_13750;
                x_13753 = x_13751;
                x_13750 = ((__local
                            int32_t *) scan_arr_mem_17064)[squot32(local_tid_17060,
                                                                   32) - 1];
                x_13751 = ((__local
                            double *) scan_arr_mem_17066)[squot32(local_tid_17060,
                                                                  32) - 1];
            }
            // perform operation
            {
                int32_t f_13754 = x_13750 | x_13752;
                bool cond_13755 = slt32(0, x_13752);
                double res_13756;
                
                if (cond_13755) {
                    res_13756 = x_13753;
                } else {
                    double res_13757 = x_13751 + x_13753;
                    
                    res_13756 = res_13757;
                }
                x_13750 = f_13754;
                x_13751 = res_13756;
            }
            // write final result
            {
                ((__local int32_t *) scan_arr_mem_17064)[local_tid_17060] =
                    x_13750;
                ((__local double *) scan_arr_mem_17066)[local_tid_17060] =
                    x_13751;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_17060, 32) == 0) {
            ((__local int32_t *) scan_arr_mem_17064)[local_tid_17060] = x_13752;
            ((__local double *) scan_arr_mem_17066)[local_tid_17060] = x_13753;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_14479, n_13426)) {
            ((__global int32_t *) mem_16479)[gtid_14479] = ((__local
                                                             int32_t *) scan_arr_mem_17064)[local_tid_17060];
            ((__global double *) mem_16482)[gtid_14479] = ((__local
                                                            double *) scan_arr_mem_17066)[local_tid_17060];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_14475
}
__kernel void tridagParFlatziscan_stage2_14489(__global int *global_failure,
                                               __local volatile
                                               int64_t *scan_arr_mem_17148_backing_aligned_0,
                                               __local volatile
                                               int64_t *scan_arr_mem_17146_backing_aligned_1,
                                               __local volatile
                                               int64_t *scan_arr_mem_17144_backing_aligned_2,
                                               int32_t n_13426, __global
                                               unsigned char *mem_16486,
                                               __global
                                               unsigned char *mem_16489,
                                               __global
                                               unsigned char *mem_16492,
                                               int32_t stage1_num_groups_17093,
                                               int32_t num_threads_17094)
{
    #define segscan_group_sizze_14484 (tridagParFlatzisegscan_group_sizze_14483)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17148_backing_2 =
                          (__local volatile
                           char *) scan_arr_mem_17148_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17146_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17146_backing_aligned_1;
    __local volatile char *restrict scan_arr_mem_17144_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17144_backing_aligned_2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17139;
    int32_t local_tid_17140;
    int32_t group_sizze_17143;
    int32_t wave_sizze_17142;
    int32_t group_tid_17141;
    
    global_tid_17139 = get_global_id(0);
    local_tid_17140 = get_local_id(0);
    group_sizze_17143 = get_local_size(0);
    wave_sizze_17142 = LOCKSTEP_WIDTH;
    group_tid_17141 = get_group_id(0);
    
    int32_t phys_tid_14489;
    
    phys_tid_14489 = global_tid_17139;
    
    __local char *scan_arr_mem_17144;
    __local char *scan_arr_mem_17146;
    __local char *scan_arr_mem_17148;
    
    scan_arr_mem_17144 = (__local char *) scan_arr_mem_17144_backing_0;
    scan_arr_mem_17146 = (__local char *) scan_arr_mem_17146_backing_1;
    scan_arr_mem_17148 = (__local char *) scan_arr_mem_17148_backing_2;
    
    int32_t flat_idx_17150;
    
    flat_idx_17150 = (local_tid_17140 + 1) * (segscan_group_sizze_14484 *
                                              sdiv_up32(n_13426,
                                                        num_threads_17094)) - 1;
    
    int32_t gtid_14488;
    
    gtid_14488 = flat_idx_17150;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_14488, n_13426)) {
            ((__local int32_t *) scan_arr_mem_17144)[local_tid_17140] =
                ((__global int32_t *) mem_16486)[gtid_14488];
            ((__local double *) scan_arr_mem_17146)[local_tid_17140] =
                ((__global double *) mem_16489)[gtid_14488];
            ((__local double *) scan_arr_mem_17148)[local_tid_17140] =
                ((__global double *) mem_16492)[gtid_14488];
        } else {
            ((__local int32_t *) scan_arr_mem_17144)[local_tid_17140] = 0;
            ((__local double *) scan_arr_mem_17146)[local_tid_17140] = 0.0;
            ((__local double *) scan_arr_mem_17148)[local_tid_17140] = 1.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t x_13774;
    double x_13775;
    double x_13776;
    int32_t x_13777;
    double x_13778;
    double x_13779;
    int32_t x_17151;
    double x_17152;
    double x_17153;
    int32_t x_17154;
    double x_17155;
    double x_17156;
    int32_t skip_threads_17164;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_17140, stage1_num_groups_17093)) {
            x_13777 = ((volatile __local
                        int32_t *) scan_arr_mem_17144)[local_tid_17140];
            x_13778 = ((volatile __local
                        double *) scan_arr_mem_17146)[local_tid_17140];
            x_13779 = ((volatile __local
                        double *) scan_arr_mem_17148)[local_tid_17140];
            if ((local_tid_17140 - squot32(local_tid_17140, 32) * 32) == 0) {
                x_13774 = x_13777;
                x_13775 = x_13778;
                x_13776 = x_13779;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_17164 = 1;
        while (slt32(skip_threads_17164, 32)) {
            if (sle32(skip_threads_17164, local_tid_17140 -
                      squot32(local_tid_17140, 32) * 32) &&
                slt32(local_tid_17140, stage1_num_groups_17093)) {
                // read operands
                {
                    x_13774 = ((volatile __local
                                int32_t *) scan_arr_mem_17144)[local_tid_17140 -
                                                               skip_threads_17164];
                    x_13775 = ((volatile __local
                                double *) scan_arr_mem_17146)[local_tid_17140 -
                                                              skip_threads_17164];
                    x_13776 = ((volatile __local
                                double *) scan_arr_mem_17148)[local_tid_17140 -
                                                              skip_threads_17164];
                }
                // perform operation
                {
                    int32_t f_13780 = x_13774 | x_13777;
                    bool cond_13781 = slt32(0, x_13777);
                    double res_13782;
                    double res_13783;
                    
                    if (cond_13781) {
                        res_13782 = x_13778;
                        res_13783 = x_13779;
                    } else {
                        double y_13784 = x_13775 * x_13779;
                        double res_13785 = x_13778 + y_13784;
                        double res_13786 = x_13776 * x_13779;
                        
                        res_13782 = res_13785;
                        res_13783 = res_13786;
                    }
                    x_13774 = f_13780;
                    x_13775 = res_13782;
                    x_13776 = res_13783;
                }
            }
            if (sle32(wave_sizze_17142, skip_threads_17164)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_17164, local_tid_17140 -
                      squot32(local_tid_17140, 32) * 32) &&
                slt32(local_tid_17140, stage1_num_groups_17093)) {
                // write result
                {
                    ((volatile __local
                      int32_t *) scan_arr_mem_17144)[local_tid_17140] = x_13774;
                    x_13777 = x_13774;
                    ((volatile __local
                      double *) scan_arr_mem_17146)[local_tid_17140] = x_13775;
                    x_13778 = x_13775;
                    ((volatile __local
                      double *) scan_arr_mem_17148)[local_tid_17140] = x_13776;
                    x_13779 = x_13776;
                }
            }
            if (sle32(wave_sizze_17142, skip_threads_17164)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_17164 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_17140 - squot32(local_tid_17140, 32) * 32) == 31 &&
            slt32(local_tid_17140, stage1_num_groups_17093)) {
            ((volatile __local
              int32_t *) scan_arr_mem_17144)[squot32(local_tid_17140, 32)] =
                x_13774;
            ((volatile __local
              double *) scan_arr_mem_17146)[squot32(local_tid_17140, 32)] =
                x_13775;
            ((volatile __local
              double *) scan_arr_mem_17148)[squot32(local_tid_17140, 32)] =
                x_13776;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_17165;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_17140, 32) == 0 && slt32(local_tid_17140,
                                                           stage1_num_groups_17093)) {
                x_17154 = ((volatile __local
                            int32_t *) scan_arr_mem_17144)[local_tid_17140];
                x_17155 = ((volatile __local
                            double *) scan_arr_mem_17146)[local_tid_17140];
                x_17156 = ((volatile __local
                            double *) scan_arr_mem_17148)[local_tid_17140];
                if ((local_tid_17140 - squot32(local_tid_17140, 32) * 32) ==
                    0) {
                    x_17151 = x_17154;
                    x_17152 = x_17155;
                    x_17153 = x_17156;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_17165 = 1;
            while (slt32(skip_threads_17165, 32)) {
                if (sle32(skip_threads_17165, local_tid_17140 -
                          squot32(local_tid_17140, 32) * 32) &&
                    (squot32(local_tid_17140, 32) == 0 && slt32(local_tid_17140,
                                                                stage1_num_groups_17093))) {
                    // read operands
                    {
                        x_17151 = ((volatile __local
                                    int32_t *) scan_arr_mem_17144)[local_tid_17140 -
                                                                   skip_threads_17165];
                        x_17152 = ((volatile __local
                                    double *) scan_arr_mem_17146)[local_tid_17140 -
                                                                  skip_threads_17165];
                        x_17153 = ((volatile __local
                                    double *) scan_arr_mem_17148)[local_tid_17140 -
                                                                  skip_threads_17165];
                    }
                    // perform operation
                    {
                        int32_t f_17157 = x_17151 | x_17154;
                        bool cond_17158 = slt32(0, x_17154);
                        double res_17159;
                        double res_17160;
                        
                        if (cond_17158) {
                            res_17159 = x_17155;
                            res_17160 = x_17156;
                        } else {
                            double y_17161 = x_17152 * x_17156;
                            double res_17162 = x_17155 + y_17161;
                            double res_17163 = x_17153 * x_17156;
                            
                            res_17159 = res_17162;
                            res_17160 = res_17163;
                        }
                        x_17151 = f_17157;
                        x_17152 = res_17159;
                        x_17153 = res_17160;
                    }
                }
                if (sle32(wave_sizze_17142, skip_threads_17165)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_17165, local_tid_17140 -
                          squot32(local_tid_17140, 32) * 32) &&
                    (squot32(local_tid_17140, 32) == 0 && slt32(local_tid_17140,
                                                                stage1_num_groups_17093))) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) scan_arr_mem_17144)[local_tid_17140] =
                            x_17151;
                        x_17154 = x_17151;
                        ((volatile __local
                          double *) scan_arr_mem_17146)[local_tid_17140] =
                            x_17152;
                        x_17155 = x_17152;
                        ((volatile __local
                          double *) scan_arr_mem_17148)[local_tid_17140] =
                            x_17153;
                        x_17156 = x_17153;
                    }
                }
                if (sle32(wave_sizze_17142, skip_threads_17165)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_17165 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_17140, 32) == 0 || !slt32(local_tid_17140,
                                                          stage1_num_groups_17093))) {
            // read operands
            {
                x_13777 = x_13774;
                x_13778 = x_13775;
                x_13779 = x_13776;
                x_13774 = ((__local
                            int32_t *) scan_arr_mem_17144)[squot32(local_tid_17140,
                                                                   32) - 1];
                x_13775 = ((__local
                            double *) scan_arr_mem_17146)[squot32(local_tid_17140,
                                                                  32) - 1];
                x_13776 = ((__local
                            double *) scan_arr_mem_17148)[squot32(local_tid_17140,
                                                                  32) - 1];
            }
            // perform operation
            {
                int32_t f_13780 = x_13774 | x_13777;
                bool cond_13781 = slt32(0, x_13777);
                double res_13782;
                double res_13783;
                
                if (cond_13781) {
                    res_13782 = x_13778;
                    res_13783 = x_13779;
                } else {
                    double y_13784 = x_13775 * x_13779;
                    double res_13785 = x_13778 + y_13784;
                    double res_13786 = x_13776 * x_13779;
                    
                    res_13782 = res_13785;
                    res_13783 = res_13786;
                }
                x_13774 = f_13780;
                x_13775 = res_13782;
                x_13776 = res_13783;
            }
            // write final result
            {
                ((__local int32_t *) scan_arr_mem_17144)[local_tid_17140] =
                    x_13774;
                ((__local double *) scan_arr_mem_17146)[local_tid_17140] =
                    x_13775;
                ((__local double *) scan_arr_mem_17148)[local_tid_17140] =
                    x_13776;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_17140, 32) == 0) {
            ((__local int32_t *) scan_arr_mem_17144)[local_tid_17140] = x_13777;
            ((__local double *) scan_arr_mem_17146)[local_tid_17140] = x_13778;
            ((__local double *) scan_arr_mem_17148)[local_tid_17140] = x_13779;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_14488, n_13426)) {
            ((__global int32_t *) mem_16486)[gtid_14488] = ((__local
                                                             int32_t *) scan_arr_mem_17144)[local_tid_17140];
            ((__global double *) mem_16489)[gtid_14488] = ((__local
                                                            double *) scan_arr_mem_17146)[local_tid_17140];
            ((__global double *) mem_16492)[gtid_14488] = ((__local
                                                            double *) scan_arr_mem_17148)[local_tid_17140];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_14484
}
__kernel void tridagParFlatziscan_stage3_14146(__global int *global_failure,
                                               int32_t n_13426,
                                               int32_t num_groups_14143,
                                               __global
                                               unsigned char *mem_16423,
                                               __global
                                               unsigned char *mem_16426,
                                               int32_t num_threads_16628,
                                               int32_t required_groups_16681)
{
    #define segscan_group_sizze_14141 (tridagParFlatzisegscan_group_sizze_14140)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16682;
    int32_t local_tid_16683;
    int32_t group_sizze_16686;
    int32_t wave_sizze_16685;
    int32_t group_tid_16684;
    
    global_tid_16682 = get_global_id(0);
    local_tid_16683 = get_local_id(0);
    group_sizze_16686 = get_local_size(0);
    wave_sizze_16685 = LOCKSTEP_WIDTH;
    group_tid_16684 = get_group_id(0);
    
    int32_t phys_tid_14146;
    
    phys_tid_14146 = global_tid_16682;
    
    int32_t phys_group_id_16687;
    
    phys_group_id_16687 = get_group_id(0);
    for (int32_t i_16688 = 0; i_16688 < sdiv_up32(required_groups_16681 -
                                                  phys_group_id_16687,
                                                  num_groups_14143);
         i_16688++) {
        int32_t virt_group_id_16689 = phys_group_id_16687 + i_16688 *
                num_groups_14143;
        int32_t flat_idx_16690 = virt_group_id_16689 *
                segscan_group_sizze_14141 + local_tid_16683;
        int32_t gtid_14145 = flat_idx_16690;
        int32_t orig_group_16691 = squot32(flat_idx_16690,
                                           segscan_group_sizze_14141 *
                                           sdiv_up32(n_13426,
                                                     num_threads_16628));
        int32_t carry_in_flat_idx_16692 = orig_group_16691 *
                (segscan_group_sizze_14141 * sdiv_up32(n_13426,
                                                       num_threads_16628)) - 1;
        
        if (slt32(gtid_14145, n_13426)) {
            if (!(orig_group_16691 == 0 || flat_idx_16690 == (orig_group_16691 +
                                                              1) *
                  (segscan_group_sizze_14141 * sdiv_up32(n_13426,
                                                         num_threads_16628)) -
                  1)) {
                int32_t x_13478;
                double x_13479;
                int32_t x_13480;
                double x_13481;
                
                x_13478 = ((__global
                            int32_t *) mem_16423)[carry_in_flat_idx_16692];
                x_13479 = ((__global
                            double *) mem_16426)[carry_in_flat_idx_16692];
                x_13480 = ((__global int32_t *) mem_16423)[gtid_14145];
                x_13481 = ((__global double *) mem_16426)[gtid_14145];
                
                int32_t f_13482;
                
                f_13482 = x_13478 | x_13480;
                
                bool cond_13483 = slt32(0, x_13480);
                double res_13484;
                
                if (cond_13483) {
                    res_13484 = x_13481;
                } else {
                    double res_13485 = x_13479 + x_13481;
                    
                    res_13484 = res_13485;
                }
                x_13478 = f_13482;
                x_13479 = res_13484;
                ((__global int32_t *) mem_16423)[gtid_14145] = x_13478;
                ((__global double *) mem_16426)[gtid_14145] = x_13479;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_14141
}
__kernel void tridagParFlatziscan_stage3_14155(__global int *global_failure,
                                               int32_t n_13426,
                                               int32_t num_groups_14152,
                                               __global
                                               unsigned char *mem_16430,
                                               __global
                                               unsigned char *mem_16433,
                                               __global
                                               unsigned char *mem_16436,
                                               __global
                                               unsigned char *mem_16439,
                                               __global
                                               unsigned char *mem_16442,
                                               int32_t num_threads_16696,
                                               int32_t required_groups_16836)
{
    #define segscan_group_sizze_14150 (tridagParFlatzisegscan_group_sizze_14149)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16837;
    int32_t local_tid_16838;
    int32_t group_sizze_16841;
    int32_t wave_sizze_16840;
    int32_t group_tid_16839;
    
    global_tid_16837 = get_global_id(0);
    local_tid_16838 = get_local_id(0);
    group_sizze_16841 = get_local_size(0);
    wave_sizze_16840 = LOCKSTEP_WIDTH;
    group_tid_16839 = get_group_id(0);
    
    int32_t phys_tid_14155;
    
    phys_tid_14155 = global_tid_16837;
    
    int32_t phys_group_id_16842;
    
    phys_group_id_16842 = get_group_id(0);
    for (int32_t i_16843 = 0; i_16843 < sdiv_up32(required_groups_16836 -
                                                  phys_group_id_16842,
                                                  num_groups_14152);
         i_16843++) {
        int32_t virt_group_id_16844 = phys_group_id_16842 + i_16843 *
                num_groups_14152;
        int32_t flat_idx_16845 = virt_group_id_16844 *
                segscan_group_sizze_14150 + local_tid_16838;
        int32_t gtid_14154 = flat_idx_16845;
        int32_t orig_group_16846 = squot32(flat_idx_16845,
                                           segscan_group_sizze_14150 *
                                           sdiv_up32(n_13426,
                                                     num_threads_16696));
        int32_t carry_in_flat_idx_16847 = orig_group_16846 *
                (segscan_group_sizze_14150 * sdiv_up32(n_13426,
                                                       num_threads_16696)) - 1;
        
        if (slt32(gtid_14154, n_13426)) {
            if (!(orig_group_16846 == 0 || flat_idx_16845 == (orig_group_16846 +
                                                              1) *
                  (segscan_group_sizze_14150 * sdiv_up32(n_13426,
                                                         num_threads_16696)) -
                  1)) {
                int32_t x_13504;
                double x_13505;
                double x_13506;
                double x_13507;
                double x_13508;
                int32_t x_13509;
                double x_13510;
                double x_13511;
                double x_13512;
                double x_13513;
                
                x_13504 = ((__global
                            int32_t *) mem_16430)[carry_in_flat_idx_16847];
                x_13505 = ((__global
                            double *) mem_16433)[carry_in_flat_idx_16847];
                x_13506 = ((__global
                            double *) mem_16436)[carry_in_flat_idx_16847];
                x_13507 = ((__global
                            double *) mem_16439)[carry_in_flat_idx_16847];
                x_13508 = ((__global
                            double *) mem_16442)[carry_in_flat_idx_16847];
                x_13509 = ((__global int32_t *) mem_16430)[gtid_14154];
                x_13510 = ((__global double *) mem_16433)[gtid_14154];
                x_13511 = ((__global double *) mem_16436)[gtid_14154];
                x_13512 = ((__global double *) mem_16439)[gtid_14154];
                x_13513 = ((__global double *) mem_16442)[gtid_14154];
                
                int32_t f_13514;
                
                f_13514 = x_13504 | x_13509;
                
                bool cond_13515 = slt32(0, x_13509);
                double res_13516;
                double res_13517;
                double res_13518;
                double res_13519;
                
                if (cond_13515) {
                    res_13516 = x_13510;
                    res_13517 = x_13511;
                    res_13518 = x_13512;
                    res_13519 = x_13513;
                } else {
                    double y_13520 = x_13505 * x_13510;
                    double value_13521 = 1.0 / y_13520;
                    double y_13522 = x_13507 * x_13511;
                    double x_13523 = y_13520 + y_13522;
                    double res_13524 = value_13521 * x_13523;
                    double x_13525 = x_13506 * x_13510;
                    double y_13526 = x_13508 * x_13511;
                    double x_13527 = x_13525 + y_13526;
                    double res_13528 = value_13521 * x_13527;
                    double x_13529 = x_13505 * x_13512;
                    double y_13530 = x_13507 * x_13513;
                    double x_13531 = x_13529 + y_13530;
                    double res_13532 = value_13521 * x_13531;
                    double x_13533 = x_13506 * x_13512;
                    double y_13534 = x_13508 * x_13513;
                    double x_13535 = x_13533 + y_13534;
                    double res_13536 = value_13521 * x_13535;
                    
                    res_13516 = res_13524;
                    res_13517 = res_13528;
                    res_13518 = res_13532;
                    res_13519 = res_13536;
                }
                x_13504 = f_13514;
                x_13505 = res_13516;
                x_13506 = res_13517;
                x_13507 = res_13518;
                x_13508 = res_13519;
                ((__global int32_t *) mem_16430)[gtid_14154] = x_13504;
                ((__global double *) mem_16433)[gtid_14154] = x_13505;
                ((__global double *) mem_16436)[gtid_14154] = x_13506;
                ((__global double *) mem_16439)[gtid_14154] = x_13507;
                ((__global double *) mem_16442)[gtid_14154] = x_13508;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_14150
}
__kernel void tridagParFlatziscan_stage3_14355(__global int *global_failure,
                                               int32_t n_13426,
                                               int32_t num_groups_14352,
                                               __global
                                               unsigned char *mem_16454,
                                               __global
                                               unsigned char *mem_16457,
                                               int32_t num_threads_16861,
                                               int32_t required_groups_16914)
{
    #define segscan_group_sizze_14350 (tridagParFlatzisegscan_group_sizze_14349)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16915;
    int32_t local_tid_16916;
    int32_t group_sizze_16919;
    int32_t wave_sizze_16918;
    int32_t group_tid_16917;
    
    global_tid_16915 = get_global_id(0);
    local_tid_16916 = get_local_id(0);
    group_sizze_16919 = get_local_size(0);
    wave_sizze_16918 = LOCKSTEP_WIDTH;
    group_tid_16917 = get_group_id(0);
    
    int32_t phys_tid_14355;
    
    phys_tid_14355 = global_tid_16915;
    
    int32_t phys_group_id_16920;
    
    phys_group_id_16920 = get_group_id(0);
    for (int32_t i_16921 = 0; i_16921 < sdiv_up32(required_groups_16914 -
                                                  phys_group_id_16920,
                                                  num_groups_14352);
         i_16921++) {
        int32_t virt_group_id_16922 = phys_group_id_16920 + i_16921 *
                num_groups_14352;
        int32_t flat_idx_16923 = virt_group_id_16922 *
                segscan_group_sizze_14350 + local_tid_16916;
        int32_t gtid_14354 = flat_idx_16923;
        int32_t orig_group_16924 = squot32(flat_idx_16923,
                                           segscan_group_sizze_14350 *
                                           sdiv_up32(n_13426,
                                                     num_threads_16861));
        int32_t carry_in_flat_idx_16925 = orig_group_16924 *
                (segscan_group_sizze_14350 * sdiv_up32(n_13426,
                                                       num_threads_16861)) - 1;
        
        if (slt32(gtid_14354, n_13426)) {
            if (!(orig_group_16924 == 0 || flat_idx_16923 == (orig_group_16924 +
                                                              1) *
                  (segscan_group_sizze_14350 * sdiv_up32(n_13426,
                                                         num_threads_16861)) -
                  1)) {
                int32_t x_13646;
                double x_13647;
                int32_t x_13648;
                double x_13649;
                
                x_13646 = ((__global
                            int32_t *) mem_16454)[carry_in_flat_idx_16925];
                x_13647 = ((__global
                            double *) mem_16457)[carry_in_flat_idx_16925];
                x_13648 = ((__global int32_t *) mem_16454)[gtid_14354];
                x_13649 = ((__global double *) mem_16457)[gtid_14354];
                
                int32_t f_13650;
                
                f_13650 = x_13646 | x_13648;
                
                bool cond_13651 = slt32(0, x_13648);
                double res_13652;
                
                if (cond_13651) {
                    res_13652 = x_13649;
                } else {
                    double res_13653 = x_13647 + x_13649;
                    
                    res_13652 = res_13653;
                }
                x_13646 = f_13650;
                x_13647 = res_13652;
                ((__global int32_t *) mem_16454)[gtid_14354] = x_13646;
                ((__global double *) mem_16457)[gtid_14354] = x_13647;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_14350
}
__kernel void tridagParFlatziscan_stage3_14364(__global int *global_failure,
                                               int32_t n_13426,
                                               int32_t num_groups_14361,
                                               __global
                                               unsigned char *mem_16461,
                                               __global
                                               unsigned char *mem_16464,
                                               __global
                                               unsigned char *mem_16467,
                                               int32_t num_threads_16929,
                                               int32_t required_groups_17001)
{
    #define segscan_group_sizze_14359 (tridagParFlatzisegscan_group_sizze_14358)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17002;
    int32_t local_tid_17003;
    int32_t group_sizze_17006;
    int32_t wave_sizze_17005;
    int32_t group_tid_17004;
    
    global_tid_17002 = get_global_id(0);
    local_tid_17003 = get_local_id(0);
    group_sizze_17006 = get_local_size(0);
    wave_sizze_17005 = LOCKSTEP_WIDTH;
    group_tid_17004 = get_group_id(0);
    
    int32_t phys_tid_14364;
    
    phys_tid_14364 = global_tid_17002;
    
    int32_t phys_group_id_17007;
    
    phys_group_id_17007 = get_group_id(0);
    for (int32_t i_17008 = 0; i_17008 < sdiv_up32(required_groups_17001 -
                                                  phys_group_id_17007,
                                                  num_groups_14361);
         i_17008++) {
        int32_t virt_group_id_17009 = phys_group_id_17007 + i_17008 *
                num_groups_14361;
        int32_t flat_idx_17010 = virt_group_id_17009 *
                segscan_group_sizze_14359 + local_tid_17003;
        int32_t gtid_14363 = flat_idx_17010;
        int32_t orig_group_17011 = squot32(flat_idx_17010,
                                           segscan_group_sizze_14359 *
                                           sdiv_up32(n_13426,
                                                     num_threads_16929));
        int32_t carry_in_flat_idx_17012 = orig_group_17011 *
                (segscan_group_sizze_14359 * sdiv_up32(n_13426,
                                                       num_threads_16929)) - 1;
        
        if (slt32(gtid_14363, n_13426)) {
            if (!(orig_group_17011 == 0 || flat_idx_17010 == (orig_group_17011 +
                                                              1) *
                  (segscan_group_sizze_14359 * sdiv_up32(n_13426,
                                                         num_threads_16929)) -
                  1)) {
                int32_t x_13670;
                double x_13671;
                double x_13672;
                int32_t x_13673;
                double x_13674;
                double x_13675;
                
                x_13670 = ((__global
                            int32_t *) mem_16461)[carry_in_flat_idx_17012];
                x_13671 = ((__global
                            double *) mem_16464)[carry_in_flat_idx_17012];
                x_13672 = ((__global
                            double *) mem_16467)[carry_in_flat_idx_17012];
                x_13673 = ((__global int32_t *) mem_16461)[gtid_14363];
                x_13674 = ((__global double *) mem_16464)[gtid_14363];
                x_13675 = ((__global double *) mem_16467)[gtid_14363];
                
                int32_t f_13676;
                
                f_13676 = x_13670 | x_13673;
                
                bool cond_13677 = slt32(0, x_13673);
                double res_13678;
                double res_13679;
                
                if (cond_13677) {
                    res_13678 = x_13674;
                    res_13679 = x_13675;
                } else {
                    double y_13680 = x_13671 * x_13675;
                    double res_13681 = x_13674 + y_13680;
                    double res_13682 = x_13672 * x_13675;
                    
                    res_13678 = res_13681;
                    res_13679 = res_13682;
                }
                x_13670 = f_13676;
                x_13671 = res_13678;
                x_13672 = res_13679;
                ((__global int32_t *) mem_16461)[gtid_14363] = x_13670;
                ((__global double *) mem_16464)[gtid_14363] = x_13671;
                ((__global double *) mem_16467)[gtid_14363] = x_13672;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_14359
}
__kernel void tridagParFlatziscan_stage3_14480(__global int *global_failure,
                                               int32_t n_13426,
                                               int32_t num_groups_14477,
                                               __global
                                               unsigned char *mem_16479,
                                               __global
                                               unsigned char *mem_16482,
                                               int32_t num_threads_17026,
                                               int32_t required_groups_17079)
{
    #define segscan_group_sizze_14475 (tridagParFlatzisegscan_group_sizze_14474)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17080;
    int32_t local_tid_17081;
    int32_t group_sizze_17084;
    int32_t wave_sizze_17083;
    int32_t group_tid_17082;
    
    global_tid_17080 = get_global_id(0);
    local_tid_17081 = get_local_id(0);
    group_sizze_17084 = get_local_size(0);
    wave_sizze_17083 = LOCKSTEP_WIDTH;
    group_tid_17082 = get_group_id(0);
    
    int32_t phys_tid_14480;
    
    phys_tid_14480 = global_tid_17080;
    
    int32_t phys_group_id_17085;
    
    phys_group_id_17085 = get_group_id(0);
    for (int32_t i_17086 = 0; i_17086 < sdiv_up32(required_groups_17079 -
                                                  phys_group_id_17085,
                                                  num_groups_14477);
         i_17086++) {
        int32_t virt_group_id_17087 = phys_group_id_17085 + i_17086 *
                num_groups_14477;
        int32_t flat_idx_17088 = virt_group_id_17087 *
                segscan_group_sizze_14475 + local_tid_17081;
        int32_t gtid_14479 = flat_idx_17088;
        int32_t orig_group_17089 = squot32(flat_idx_17088,
                                           segscan_group_sizze_14475 *
                                           sdiv_up32(n_13426,
                                                     num_threads_17026));
        int32_t carry_in_flat_idx_17090 = orig_group_17089 *
                (segscan_group_sizze_14475 * sdiv_up32(n_13426,
                                                       num_threads_17026)) - 1;
        
        if (slt32(gtid_14479, n_13426)) {
            if (!(orig_group_17089 == 0 || flat_idx_17088 == (orig_group_17089 +
                                                              1) *
                  (segscan_group_sizze_14475 * sdiv_up32(n_13426,
                                                         num_threads_17026)) -
                  1)) {
                int32_t x_13750;
                double x_13751;
                int32_t x_13752;
                double x_13753;
                
                x_13750 = ((__global
                            int32_t *) mem_16479)[carry_in_flat_idx_17090];
                x_13751 = ((__global
                            double *) mem_16482)[carry_in_flat_idx_17090];
                x_13752 = ((__global int32_t *) mem_16479)[gtid_14479];
                x_13753 = ((__global double *) mem_16482)[gtid_14479];
                
                int32_t f_13754;
                
                f_13754 = x_13750 | x_13752;
                
                bool cond_13755 = slt32(0, x_13752);
                double res_13756;
                
                if (cond_13755) {
                    res_13756 = x_13753;
                } else {
                    double res_13757 = x_13751 + x_13753;
                    
                    res_13756 = res_13757;
                }
                x_13750 = f_13754;
                x_13751 = res_13756;
                ((__global int32_t *) mem_16479)[gtid_14479] = x_13750;
                ((__global double *) mem_16482)[gtid_14479] = x_13751;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_14475
}
__kernel void tridagParFlatziscan_stage3_14489(__global int *global_failure,
                                               int32_t n_13426,
                                               int32_t num_groups_14486,
                                               __global
                                               unsigned char *mem_16486,
                                               __global
                                               unsigned char *mem_16489,
                                               __global
                                               unsigned char *mem_16492,
                                               int32_t num_threads_17094,
                                               int32_t required_groups_17166)
{
    #define segscan_group_sizze_14484 (tridagParFlatzisegscan_group_sizze_14483)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17167;
    int32_t local_tid_17168;
    int32_t group_sizze_17171;
    int32_t wave_sizze_17170;
    int32_t group_tid_17169;
    
    global_tid_17167 = get_global_id(0);
    local_tid_17168 = get_local_id(0);
    group_sizze_17171 = get_local_size(0);
    wave_sizze_17170 = LOCKSTEP_WIDTH;
    group_tid_17169 = get_group_id(0);
    
    int32_t phys_tid_14489;
    
    phys_tid_14489 = global_tid_17167;
    
    int32_t phys_group_id_17172;
    
    phys_group_id_17172 = get_group_id(0);
    for (int32_t i_17173 = 0; i_17173 < sdiv_up32(required_groups_17166 -
                                                  phys_group_id_17172,
                                                  num_groups_14486);
         i_17173++) {
        int32_t virt_group_id_17174 = phys_group_id_17172 + i_17173 *
                num_groups_14486;
        int32_t flat_idx_17175 = virt_group_id_17174 *
                segscan_group_sizze_14484 + local_tid_17168;
        int32_t gtid_14488 = flat_idx_17175;
        int32_t orig_group_17176 = squot32(flat_idx_17175,
                                           segscan_group_sizze_14484 *
                                           sdiv_up32(n_13426,
                                                     num_threads_17094));
        int32_t carry_in_flat_idx_17177 = orig_group_17176 *
                (segscan_group_sizze_14484 * sdiv_up32(n_13426,
                                                       num_threads_17094)) - 1;
        
        if (slt32(gtid_14488, n_13426)) {
            if (!(orig_group_17176 == 0 || flat_idx_17175 == (orig_group_17176 +
                                                              1) *
                  (segscan_group_sizze_14484 * sdiv_up32(n_13426,
                                                         num_threads_17094)) -
                  1)) {
                int32_t x_13774;
                double x_13775;
                double x_13776;
                int32_t x_13777;
                double x_13778;
                double x_13779;
                
                x_13774 = ((__global
                            int32_t *) mem_16486)[carry_in_flat_idx_17177];
                x_13775 = ((__global
                            double *) mem_16489)[carry_in_flat_idx_17177];
                x_13776 = ((__global
                            double *) mem_16492)[carry_in_flat_idx_17177];
                x_13777 = ((__global int32_t *) mem_16486)[gtid_14488];
                x_13778 = ((__global double *) mem_16489)[gtid_14488];
                x_13779 = ((__global double *) mem_16492)[gtid_14488];
                
                int32_t f_13780;
                
                f_13780 = x_13774 | x_13777;
                
                bool cond_13781 = slt32(0, x_13777);
                double res_13782;
                double res_13783;
                
                if (cond_13781) {
                    res_13782 = x_13778;
                    res_13783 = x_13779;
                } else {
                    double y_13784 = x_13775 * x_13779;
                    double res_13785 = x_13778 + y_13784;
                    double res_13786 = x_13776 * x_13779;
                    
                    res_13782 = res_13785;
                    res_13783 = res_13786;
                }
                x_13774 = f_13780;
                x_13775 = res_13782;
                x_13776 = res_13783;
                ((__global int32_t *) mem_16486)[gtid_14488] = x_13774;
                ((__global double *) mem_16489)[gtid_14488] = x_13775;
                ((__global double *) mem_16492)[gtid_14488] = x_13776;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_14484
}
__kernel void tridagParFlatzisegmap_14130(__global int *global_failure,
                                          int32_t n_13426,
                                          int32_t segSizze_13434,
                                          int32_t segCount_13435, __global
                                          unsigned char *b_mem_16410, __global
                                          unsigned char *mem_16415, __global
                                          unsigned char *mem_16418)
{
    #define segmap_group_sizze_14134 (tridagParFlatzisegmap_group_sizze_14133)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16620;
    int32_t local_tid_16621;
    int32_t group_sizze_16624;
    int32_t wave_sizze_16623;
    int32_t group_tid_16622;
    
    global_tid_16620 = get_global_id(0);
    local_tid_16621 = get_local_id(0);
    group_sizze_16624 = get_local_size(0);
    wave_sizze_16623 = LOCKSTEP_WIDTH;
    group_tid_16622 = get_group_id(0);
    
    int32_t phys_tid_14130;
    
    phys_tid_14130 = global_tid_16620;
    
    int32_t write_i_14129;
    
    write_i_14129 = group_tid_16622 * segmap_group_sizze_14134 +
        local_tid_16621;
    if (slt32(write_i_14129, segCount_13435)) {
        int32_t index_primexp_16393 = mul32(segSizze_13434, write_i_14129);
        double res_13452 = ((__global
                             double *) b_mem_16410)[index_primexp_16393];
        
        if (sle32(0, index_primexp_16393) && slt32(index_primexp_16393,
                                                   n_13426)) {
            ((__global double *) mem_16418)[index_primexp_16393] = res_13452;
        }
        if (sle32(0, index_primexp_16393) && slt32(index_primexp_16393,
                                                   n_13426)) {
            ((__global int32_t *) mem_16415)[index_primexp_16393] = 1;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_14134
}
__kernel void tridagParFlatzisegmap_14249(__global int *global_failure,
                                          int32_t n_13426, __global
                                          unsigned char *mem_16426, __global
                                          unsigned char *mem_16430, __global
                                          unsigned char *mem_16433, __global
                                          unsigned char *mem_16436, __global
                                          unsigned char *mem_16439, __global
                                          unsigned char *mem_16442, __global
                                          unsigned char *mem_16446)
{
    #define segmap_group_sizze_14296 (tridagParFlatzisegmap_group_sizze_14252)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16848;
    int32_t local_tid_16849;
    int32_t group_sizze_16852;
    int32_t wave_sizze_16851;
    int32_t group_tid_16850;
    
    global_tid_16848 = get_global_id(0);
    local_tid_16849 = get_local_id(0);
    group_sizze_16852 = get_local_size(0);
    wave_sizze_16851 = LOCKSTEP_WIDTH;
    group_tid_16850 = get_group_id(0);
    
    int32_t phys_tid_14249;
    
    phys_tid_14249 = global_tid_16848;
    
    int32_t gtid_14248;
    
    gtid_14248 = group_tid_16850 * segmap_group_sizze_14296 + local_tid_16849;
    if (slt32(gtid_14248, n_13426)) {
        double x_14302 = ((__global double *) mem_16426)[gtid_14248];
        int32_t x_14303 = ((__global int32_t *) mem_16430)[gtid_14248];
        double x_14304 = ((__global double *) mem_16433)[gtid_14248];
        double x_14305 = ((__global double *) mem_16436)[gtid_14248];
        double x_14306 = ((__global double *) mem_16439)[gtid_14248];
        double x_14307 = ((__global double *) mem_16442)[gtid_14248];
        bool cond_14311 = slt32(0, x_14303);
        double res_14312;
        double res_14313;
        double res_14314;
        double res_14315;
        
        if (cond_14311) {
            res_14312 = x_14304;
            res_14313 = x_14305;
            res_14314 = x_14306;
            res_14315 = x_14307;
        } else {
            double value_14317 = 1.0 / x_14304;
            double res_14320 = x_14304 * value_14317;
            double res_14324 = x_14305 * value_14317;
            double res_14328 = x_14306 * value_14317;
            double res_14332 = x_14307 * value_14317;
            
            res_14312 = res_14320;
            res_14313 = res_14324;
            res_14314 = res_14328;
            res_14315 = res_14332;
        }
        
        double x_14333 = x_14302 * res_14312;
        double x_14334 = res_14313 + x_14333;
        double x_14335 = x_14302 * res_14314;
        double y_14336 = res_14315 + x_14335;
        double res_14337 = x_14334 / y_14336;
        
        ((__global double *) mem_16446)[gtid_14248] = res_14337;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_14296
}
__kernel void tridagParFlatzisegmap_14339(__global int *global_failure,
                                          int32_t n_13426,
                                          int32_t segSizze_13434,
                                          int32_t segCount_13435, __global
                                          unsigned char *y_mem_16412, __global
                                          unsigned char *mem_16449)
{
    #define segmap_group_sizze_14343 (tridagParFlatzisegmap_group_sizze_14342)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
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
    
    int32_t phys_tid_14339;
    
    phys_tid_14339 = global_tid_16853;
    
    int32_t write_i_14338;
    
    write_i_14338 = group_tid_16855 * segmap_group_sizze_14343 +
        local_tid_16854;
    if (slt32(write_i_14338, segCount_13435)) {
        int32_t index_primexp_16394 = mul32(segSizze_13434, write_i_14338);
        double res_13625 = ((__global
                             double *) y_mem_16412)[index_primexp_16394];
        
        if (sle32(0, index_primexp_16394) && slt32(index_primexp_16394,
                                                   n_13426)) {
            ((__global double *) mem_16449)[index_primexp_16394] = res_13625;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_14343
}
__kernel void tridagParFlatzisegmap_14416(__global int *global_failure,
                                          int32_t n_13426, __global
                                          unsigned char *mem_16457, __global
                                          unsigned char *mem_16464, __global
                                          unsigned char *mem_16467, __global
                                          unsigned char *mem_16471)
{
    #define segmap_group_sizze_14442 (tridagParFlatzisegmap_group_sizze_14419)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17013;
    int32_t local_tid_17014;
    int32_t group_sizze_17017;
    int32_t wave_sizze_17016;
    int32_t group_tid_17015;
    
    global_tid_17013 = get_global_id(0);
    local_tid_17014 = get_local_id(0);
    group_sizze_17017 = get_local_size(0);
    wave_sizze_17016 = LOCKSTEP_WIDTH;
    group_tid_17015 = get_group_id(0);
    
    int32_t phys_tid_14416;
    
    phys_tid_14416 = global_tid_17013;
    
    int32_t gtid_14415;
    
    gtid_14415 = group_tid_17015 * segmap_group_sizze_14442 + local_tid_17014;
    if (slt32(gtid_14415, n_13426)) {
        double x_14448 = ((__global double *) mem_16457)[gtid_14415];
        double x_14450 = ((__global double *) mem_16464)[gtid_14415];
        double x_14451 = ((__global double *) mem_16467)[gtid_14415];
        double y_14461 = x_14448 * x_14451;
        double res_14462 = x_14450 + y_14461;
        
        ((__global double *) mem_16471)[gtid_14415] = res_14462;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_14442
}
__kernel void tridagParFlatzisegmap_14464(__global int *global_failure,
                                          int32_t n_13426,
                                          int32_t segSizze_13434,
                                          int32_t segCount_13435, __global
                                          unsigned char *mem_16446, __global
                                          unsigned char *mem_16471, __global
                                          unsigned char *mem_16474)
{
    #define segmap_group_sizze_14468 (tridagParFlatzisegmap_group_sizze_14467)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17018;
    int32_t local_tid_17019;
    int32_t group_sizze_17022;
    int32_t wave_sizze_17021;
    int32_t group_tid_17020;
    
    global_tid_17018 = get_global_id(0);
    local_tid_17019 = get_local_id(0);
    group_sizze_17022 = get_local_size(0);
    wave_sizze_17021 = LOCKSTEP_WIDTH;
    group_tid_17020 = get_group_id(0);
    
    int32_t phys_tid_14464;
    
    phys_tid_14464 = global_tid_17018;
    
    int32_t write_i_14463;
    
    write_i_14463 = group_tid_17020 * segmap_group_sizze_14468 +
        local_tid_17019;
    if (slt32(write_i_14463, segCount_13435)) {
        int32_t index_primexp_16395 = mul32(segSizze_13434, write_i_14463);
        int32_t x_13728 = add32(segSizze_13434, index_primexp_16395);
        int32_t i_13729 = sub32(x_13728, 1);
        double x_13730 = ((__global double *) mem_16471)[i_13729];
        double y_13731 = ((__global double *) mem_16446)[i_13729];
        double res_13732 = x_13730 / y_13731;
        
        if (sle32(0, index_primexp_16395) && slt32(index_primexp_16395,
                                                   n_13426)) {
            ((__global double *) mem_16474)[index_primexp_16395] = res_13732;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_14468
}
__kernel void tridagParFlatzisegmap_14541(__global int *global_failure,
                                          int32_t n_13426, __global
                                          unsigned char *mem_16482, __global
                                          unsigned char *mem_16489, __global
                                          unsigned char *mem_16492, __global
                                          unsigned char *mem_16496)
{
    #define segmap_group_sizze_14567 (tridagParFlatzisegmap_group_sizze_14544)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17178;
    int32_t local_tid_17179;
    int32_t group_sizze_17182;
    int32_t wave_sizze_17181;
    int32_t group_tid_17180;
    
    global_tid_17178 = get_global_id(0);
    local_tid_17179 = get_local_id(0);
    group_sizze_17182 = get_local_size(0);
    wave_sizze_17181 = LOCKSTEP_WIDTH;
    group_tid_17180 = get_group_id(0);
    
    int32_t phys_tid_14541;
    
    phys_tid_14541 = global_tid_17178;
    
    int32_t gtid_14540;
    
    gtid_14540 = group_tid_17180 * segmap_group_sizze_14567 + local_tid_17179;
    if (slt32(gtid_14540, n_13426)) {
        double x_14573 = ((__global double *) mem_16482)[gtid_14540];
        double x_14575 = ((__global double *) mem_16489)[gtid_14540];
        double x_14576 = ((__global double *) mem_16492)[gtid_14540];
        double y_14586 = x_14573 * x_14576;
        double res_14587 = x_14575 + y_14586;
        
        ((__global double *) mem_16496)[gtid_14540] = res_14587;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_14567
}
__kernel void tridagParFlatzisegmap_14623(__global int *global_failure,
                                          int32_t n_13426,
                                          int32_t segSizze_13434, __global
                                          unsigned char *mem_16496, __global
                                          unsigned char *mem_16500)
{
    #define segmap_group_sizze_14641 (tridagParFlatzisegmap_group_sizze_14626)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17183;
    int32_t local_tid_17184;
    int32_t group_sizze_17187;
    int32_t wave_sizze_17186;
    int32_t group_tid_17185;
    
    global_tid_17183 = get_global_id(0);
    local_tid_17184 = get_local_id(0);
    group_sizze_17187 = get_local_size(0);
    wave_sizze_17186 = LOCKSTEP_WIDTH;
    group_tid_17185 = get_group_id(0);
    
    int32_t phys_tid_14623;
    
    phys_tid_14623 = global_tid_17183;
    
    int32_t gtid_14622;
    
    gtid_14622 = group_tid_17185 * segmap_group_sizze_14641 + local_tid_17184;
    if (slt32(gtid_14622, n_13426)) {
        int32_t seg_14647 = sdiv32(gtid_14622, segSizze_13434);
        int32_t segInd_14648 = smod32(gtid_14622, segSizze_13434);
        int32_t x_14649 = mul32(segSizze_13434, seg_14647);
        int32_t x_14650 = add32(segSizze_13434, x_14649);
        int32_t x_14651 = sub32(x_14650, segInd_14648);
        int32_t i_14652 = sub32(x_14651, 1);
        double res_14653 = ((__global double *) mem_16496)[i_14652];
        
        ((__global double *) mem_16500)[gtid_14622] = res_14653;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_14641
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
                                   ["[][]f64"]), "tridagParFlat": (["[]f64",
                                                                    "[]f64",
                                                                    "[]f64",
                                                                    "[]f64",
                                                                    "i32",
                                                                    "i32"],
                                                                   ["[]f64"])}
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
    self.global_failure_args_max = 0
    self.failure_msgs=[]
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
                                       required_types=["i32", "f64", "bool"],
                                       user_sizes=sizes,
                                       all_sizes={"builtin#replicate_f64.group_size_16618": {"class": "group_size",
                                                                                   "value": None},
                                        "builtin#replicate_i32.group_size_16609": {"class": "group_size",
                                                                                   "value": None},
                                        "tridagNested.segmap_group_size_15404": {"class": "group_size", "value": None},
                                        "tridagNested.segmap_group_size_15478": {"class": "group_size", "value": None},
                                        "tridagNested.segmap_group_size_15570": {"class": "group_size", "value": None},
                                        "tridagNested.segmap_group_size_15633": {"class": "group_size", "value": None},
                                        "tridagNested.segmap_group_size_15808": {"class": "group_size", "value": None},
                                        "tridagNested.segmap_num_groups_15572": {"class": "num_groups", "value": None},
                                        "tridagNested.segscan_group_size_15536": {"class": "group_size",
                                                                                  "value": None},
                                        "tridagNested.segscan_group_size_15691": {"class": "group_size",
                                                                                  "value": None},
                                        "tridagNested.segscan_group_size_15924": {"class": "group_size",
                                                                                  "value": None},
                                        "tridagNested.segscan_num_groups_15538": {"class": "num_groups",
                                                                                  "value": None},
                                        "tridagNested.segscan_num_groups_15693": {"class": "num_groups",
                                                                                  "value": None},
                                        "tridagNested.segscan_num_groups_15926": {"class": "num_groups",
                                                                                  "value": None},
                                        "tridagNested.suff_intra_par_1": {"class": "threshold ()", "value": 32},
                                        "tridagParFlat.segmap_group_size_14133": {"class": "group_size",
                                                                                  "value": None},
                                        "tridagParFlat.segmap_group_size_14252": {"class": "group_size",
                                                                                  "value": None},
                                        "tridagParFlat.segmap_group_size_14342": {"class": "group_size",
                                                                                  "value": None},
                                        "tridagParFlat.segmap_group_size_14419": {"class": "group_size",
                                                                                  "value": None},
                                        "tridagParFlat.segmap_group_size_14467": {"class": "group_size",
                                                                                  "value": None},
                                        "tridagParFlat.segmap_group_size_14544": {"class": "group_size",
                                                                                  "value": None},
                                        "tridagParFlat.segmap_group_size_14626": {"class": "group_size",
                                                                                  "value": None},
                                        "tridagParFlat.segscan_group_size_14140": {"class": "group_size",
                                                                                   "value": None},
                                        "tridagParFlat.segscan_group_size_14149": {"class": "group_size",
                                                                                   "value": None},
                                        "tridagParFlat.segscan_group_size_14349": {"class": "group_size",
                                                                                   "value": None},
                                        "tridagParFlat.segscan_group_size_14358": {"class": "group_size",
                                                                                   "value": None},
                                        "tridagParFlat.segscan_group_size_14474": {"class": "group_size",
                                                                                   "value": None},
                                        "tridagParFlat.segscan_group_size_14483": {"class": "group_size",
                                                                                   "value": None},
                                        "tridagParFlat.segscan_num_groups_14142": {"class": "num_groups",
                                                                                   "value": None},
                                        "tridagParFlat.segscan_num_groups_14151": {"class": "num_groups",
                                                                                   "value": None},
                                        "tridagParFlat.segscan_num_groups_14351": {"class": "num_groups",
                                                                                   "value": None},
                                        "tridagParFlat.segscan_num_groups_14360": {"class": "num_groups",
                                                                                   "value": None},
                                        "tridagParFlat.segscan_num_groups_14476": {"class": "num_groups",
                                                                                   "value": None},
                                        "tridagParFlat.segscan_num_groups_14485": {"class": "num_groups",
                                                                                   "value": None}})
    self.builtinzhreplicate_f64zireplicate_16615_var = program.builtinzhreplicate_f64zireplicate_16615
    self.builtinzhreplicate_i32zireplicate_16606_var = program.builtinzhreplicate_i32zireplicate_16606
    self.tridagNestedziscan_stage1_15542_var = program.tridagNestedziscan_stage1_15542
    self.tridagNestedziscan_stage1_15697_var = program.tridagNestedziscan_stage1_15697
    self.tridagNestedziscan_stage1_15930_var = program.tridagNestedziscan_stage1_15930
    self.tridagNestedziscan_stage2_15542_var = program.tridagNestedziscan_stage2_15542
    self.tridagNestedziscan_stage2_15697_var = program.tridagNestedziscan_stage2_15697
    self.tridagNestedziscan_stage2_15930_var = program.tridagNestedziscan_stage2_15930
    self.tridagNestedziscan_stage3_15542_var = program.tridagNestedziscan_stage3_15542
    self.tridagNestedziscan_stage3_15697_var = program.tridagNestedziscan_stage3_15697
    self.tridagNestedziscan_stage3_15930_var = program.tridagNestedziscan_stage3_15930
    self.tridagNestedzisegmap_15399_var = program.tridagNestedzisegmap_15399
    self.tridagNestedzisegmap_15473_var = program.tridagNestedzisegmap_15473
    self.tridagNestedzisegmap_15567_var = program.tridagNestedzisegmap_15567
    self.tridagNestedzisegmap_15628_var = program.tridagNestedzisegmap_15628
    self.tridagNestedzisegmap_15803_var = program.tridagNestedzisegmap_15803
    self.tridagNestedzisegmap_intragroup_14717_var = program.tridagNestedzisegmap_intragroup_14717
    self.tridagParFlatziscan_stage1_14146_var = program.tridagParFlatziscan_stage1_14146
    self.tridagParFlatziscan_stage1_14155_var = program.tridagParFlatziscan_stage1_14155
    self.tridagParFlatziscan_stage1_14355_var = program.tridagParFlatziscan_stage1_14355
    self.tridagParFlatziscan_stage1_14364_var = program.tridagParFlatziscan_stage1_14364
    self.tridagParFlatziscan_stage1_14480_var = program.tridagParFlatziscan_stage1_14480
    self.tridagParFlatziscan_stage1_14489_var = program.tridagParFlatziscan_stage1_14489
    self.tridagParFlatziscan_stage2_14146_var = program.tridagParFlatziscan_stage2_14146
    self.tridagParFlatziscan_stage2_14155_var = program.tridagParFlatziscan_stage2_14155
    self.tridagParFlatziscan_stage2_14355_var = program.tridagParFlatziscan_stage2_14355
    self.tridagParFlatziscan_stage2_14364_var = program.tridagParFlatziscan_stage2_14364
    self.tridagParFlatziscan_stage2_14480_var = program.tridagParFlatziscan_stage2_14480
    self.tridagParFlatziscan_stage2_14489_var = program.tridagParFlatziscan_stage2_14489
    self.tridagParFlatziscan_stage3_14146_var = program.tridagParFlatziscan_stage3_14146
    self.tridagParFlatziscan_stage3_14155_var = program.tridagParFlatziscan_stage3_14155
    self.tridagParFlatziscan_stage3_14355_var = program.tridagParFlatziscan_stage3_14355
    self.tridagParFlatziscan_stage3_14364_var = program.tridagParFlatziscan_stage3_14364
    self.tridagParFlatziscan_stage3_14480_var = program.tridagParFlatziscan_stage3_14480
    self.tridagParFlatziscan_stage3_14489_var = program.tridagParFlatziscan_stage3_14489
    self.tridagParFlatzisegmap_14130_var = program.tridagParFlatzisegmap_14130
    self.tridagParFlatzisegmap_14249_var = program.tridagParFlatzisegmap_14249
    self.tridagParFlatzisegmap_14339_var = program.tridagParFlatzisegmap_14339
    self.tridagParFlatzisegmap_14416_var = program.tridagParFlatzisegmap_14416
    self.tridagParFlatzisegmap_14464_var = program.tridagParFlatzisegmap_14464
    self.tridagParFlatzisegmap_14541_var = program.tridagParFlatzisegmap_14541
    self.tridagParFlatzisegmap_14623_var = program.tridagParFlatzisegmap_14623
    self.constants = {}
  def futhark_builtinzhreplicate_f64(self, mem_16611, num_elems_16612,
                                     val_16613):
    group_sizze_16618 = self.sizes["builtin#replicate_f64.group_size_16618"]
    num_groups_16619 = sdiv_up32(num_elems_16612, group_sizze_16618)
    if ((1 * (np.long(num_groups_16619) * np.long(group_sizze_16618))) != 0):
      self.builtinzhreplicate_f64zireplicate_16615_var.set_args(mem_16611,
                                                                np.int32(num_elems_16612),
                                                                np.float64(val_16613))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.builtinzhreplicate_f64zireplicate_16615_var,
                                 ((np.long(num_groups_16619) * np.long(group_sizze_16618)),),
                                 (np.long(group_sizze_16618),))
      if synchronous:
        sync(self)
    return ()
  def futhark_builtinzhreplicate_i32(self, mem_16602, num_elems_16603,
                                     val_16604):
    group_sizze_16609 = self.sizes["builtin#replicate_i32.group_size_16609"]
    num_groups_16610 = sdiv_up32(num_elems_16603, group_sizze_16609)
    if ((1 * (np.long(num_groups_16610) * np.long(group_sizze_16609))) != 0):
      self.builtinzhreplicate_i32zireplicate_16606_var.set_args(mem_16602,
                                                                np.int32(num_elems_16603),
                                                                np.int32(val_16604))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.builtinzhreplicate_i32zireplicate_16606_var,
                                 ((np.long(num_groups_16610) * np.long(group_sizze_16609)),),
                                 (np.long(group_sizze_16609),))
      if synchronous:
        sync(self)
    return ()
  def futhark_tridagNested(self, a_mem_16409, b_mem_16410, c_mem_16411,
                           y_mem_16412, n_13843, m_13844, n_13845, m_13846,
                           n_13847, m_13848, n_13849, m_13850):
    dim_match_13855 = (n_13843 == n_13845)
    dim_match_13856 = (m_13844 == m_13846)
    match_13857 = (dim_match_13855 and dim_match_13856)
    empty_or_match_cert_13858 = True
    assert match_13857, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:107:1-108:49\n" % ("function arguments of wrong shape",))
    dim_match_13860 = (n_13843 == n_13847)
    dim_match_13861 = (m_13844 == m_13848)
    match_13862 = (dim_match_13860 and dim_match_13861)
    empty_or_match_cert_13863 = True
    assert match_13862, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:107:1-108:49\n" % ("function arguments of wrong shape",))
    dim_match_13865 = (n_13843 == n_13849)
    dim_match_13866 = (m_13844 == m_13850)
    match_13867 = (dim_match_13865 and dim_match_13866)
    empty_or_match_cert_13868 = True
    assert match_13867, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:107:1-108:49\n" % ("function arguments of wrong shape",))
    i_13871 = (m_13844 - np.int32(1))
    max_group_sizze_15159 = self.max_group_size
    fits_15160 = sle32(m_13844, max_group_sizze_15159)
    suff_intra_par_15158 = (self.sizes["tridagNested.suff_intra_par_1"] <= m_13844)
    intra_suff_and_fits_15161 = (suff_intra_par_15158 and fits_15160)
    m_16001 = sext_i32_i64(m_13844)
    n_16002 = sext_i32_i64(n_13843)
    nest_sizze_16004 = (m_16001 * n_16002)
    segscan_group_sizze_16005 = self.sizes["tridagNested.segscan_group_size_15924"]
    max_num_groups_16603 = self.sizes["tridagNested.segscan_num_groups_15926"]
    num_groups_16006 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(nest_sizze_16004,
                                                            sext_i32_i64(segscan_group_sizze_16005)),
                                                  sext_i32_i64(max_num_groups_16603))))
    segmap_group_sizze_16109 = self.sizes["tridagNested.segmap_group_size_15808"]
    segmap_group_sizze_16110 = sext_i32_i64(segmap_group_sizze_16109)
    segscan_group_sizze_16173 = self.sizes["tridagNested.segscan_group_size_15691"]
    max_num_groups_16604 = self.sizes["tridagNested.segscan_num_groups_15693"]
    num_groups_16174 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(nest_sizze_16004,
                                                            sext_i32_i64(segscan_group_sizze_16173)),
                                                  sext_i32_i64(max_num_groups_16604))))
    segmap_group_sizze_16233 = self.sizes["tridagNested.segmap_group_size_15633"]
    segmap_group_sizze_16234 = sext_i32_i64(segmap_group_sizze_16233)
    segmap_group_sizze_16260 = self.sizes["tridagNested.segmap_group_size_15570"]
    max_num_groups_16605 = self.sizes["tridagNested.segmap_num_groups_15572"]
    num_groups_16261 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(n_16002,
                                                            sext_i32_i64(segmap_group_sizze_16260)),
                                                  sext_i32_i64(max_num_groups_16605))))
    segscan_group_sizze_16278 = self.sizes["tridagNested.segscan_group_size_15536"]
    max_num_groups_16606 = self.sizes["tridagNested.segscan_num_groups_15538"]
    num_groups_16279 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(nest_sizze_16004,
                                                            sext_i32_i64(segscan_group_sizze_16278)),
                                                  sext_i32_i64(max_num_groups_16606))))
    segmap_group_sizze_16340 = self.sizes["tridagNested.segmap_group_size_15478"]
    segmap_group_sizze_16341 = sext_i32_i64(segmap_group_sizze_16340)
    segmap_group_sizze_16380 = self.sizes["tridagNested.segmap_group_size_15404"]
    segmap_group_sizze_16381 = sext_i32_i64(segmap_group_sizze_16380)
    binop_x_16460 = (m_16001 * n_16002)
    bytes_16457 = (np.int64(8) * binop_x_16460)
    bytes_16415 = (np.int64(8) * m_16001)
    bytes_16507 = (np.int64(8) * n_16002)
    local_memory_capacity_16978 = self.max_local_memory
    if (sle64((((((((((((bytes_16415 + bytes_16415) + bytes_16415) + bytes_16415) + bytes_16415) + bytes_16415) + bytes_16415) + bytes_16415) + bytes_16415) + bytes_16415) + bytes_16415) + bytes_16415),
              sext_i32_i64(local_memory_capacity_16978)) and intra_suff_and_fits_15161):
      mem_16461 = opencl_alloc(self, bytes_16457, "mem_16461")
      if ((1 * (np.long(n_13843) * np.long(m_13844))) != 0):
        self.tridagNestedzisegmap_intragroup_14717_var.set_args(self.global_failure,
                                                                cl.LocalMemory(np.long(bytes_16415)),
                                                                cl.LocalMemory(np.long(bytes_16415)),
                                                                cl.LocalMemory(np.long(bytes_16415)),
                                                                cl.LocalMemory(np.long(bytes_16415)),
                                                                cl.LocalMemory(np.long(bytes_16415)),
                                                                cl.LocalMemory(np.long(bytes_16415)),
                                                                cl.LocalMemory(np.long(bytes_16415)),
                                                                cl.LocalMemory(np.long(bytes_16415)),
                                                                cl.LocalMemory(np.long(bytes_16415)),
                                                                cl.LocalMemory(np.long(bytes_16415)),
                                                                cl.LocalMemory(np.long(bytes_16415)),
                                                                cl.LocalMemory(np.long(bytes_16415)),
                                                                np.int32(m_13844),
                                                                np.int32(m_13846),
                                                                np.int32(m_13848),
                                                                np.int32(m_13850),
                                                                np.int32(i_13871),
                                                                a_mem_16409,
                                                                b_mem_16410,
                                                                c_mem_16411,
                                                                y_mem_16412,
                                                                mem_16461)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedzisegmap_intragroup_14717_var,
                                   ((np.long(n_13843) * np.long(m_13844)),),
                                   (np.long(m_13844),))
        if synchronous:
          sync(self)
      res_mem_16533 = mem_16461
    else:
      mem_16467 = opencl_alloc(self, bytes_16457, "mem_16467")
      mem_16472 = opencl_alloc(self, bytes_16457, "mem_16472")
      mem_16477 = opencl_alloc(self, bytes_16457, "mem_16477")
      mem_16482 = opencl_alloc(self, bytes_16457, "mem_16482")
      if slt32(np.int32(0), (n_13843 * m_13844)):
        stage1_max_num_groups_16678 = self.max_group_size
        stage1_num_groups_16679 = smin32(stage1_max_num_groups_16678,
                                         num_groups_16006)
        num_threads_16680 = (stage1_num_groups_16679 * segscan_group_sizze_16005)
        if ((1 * (np.long(stage1_num_groups_16679) * np.long(segscan_group_sizze_16005))) != 0):
          self.tridagNestedziscan_stage1_15930_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_16005))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_16005))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_16005))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_16005))))),
                                                            np.int32(n_13843),
                                                            np.int32(m_13844),
                                                            np.int32(m_13846),
                                                            np.int32(m_13848),
                                                            a_mem_16409,
                                                            b_mem_16410,
                                                            c_mem_16411,
                                                            mem_16467,
                                                            mem_16472,
                                                            mem_16477,
                                                            mem_16482,
                                                            np.int32(num_threads_16680))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage1_15930_var,
                                     ((np.long(stage1_num_groups_16679) * np.long(segscan_group_sizze_16005)),),
                                     (np.long(segscan_group_sizze_16005),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_16679))) != 0):
          self.tridagNestedziscan_stage2_15930_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16679))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16679))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16679))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16679))))),
                                                            np.int32(n_13843),
                                                            np.int32(m_13844),
                                                            mem_16467,
                                                            mem_16472,
                                                            mem_16477,
                                                            mem_16482,
                                                            np.int32(stage1_num_groups_16679),
                                                            np.int32(num_threads_16680))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage2_15930_var,
                                     ((np.long(np.int32(1)) * np.long(stage1_num_groups_16679)),),
                                     (np.long(stage1_num_groups_16679),))
          if synchronous:
            sync(self)
        required_groups_16798 = sdiv_up32((n_13843 * m_13844),
                                          segscan_group_sizze_16005)
        if ((1 * (np.long(num_groups_16006) * np.long(segscan_group_sizze_16005))) != 0):
          self.tridagNestedziscan_stage3_15930_var.set_args(self.global_failure,
                                                            np.int32(n_13843),
                                                            np.int32(m_13844),
                                                            np.int32(num_groups_16006),
                                                            mem_16467,
                                                            mem_16472,
                                                            mem_16477,
                                                            mem_16482,
                                                            np.int32(num_threads_16680),
                                                            np.int32(required_groups_16798))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage3_15930_var,
                                     ((np.long(num_groups_16006) * np.long(segscan_group_sizze_16005)),),
                                     (np.long(segscan_group_sizze_16005),))
          if synchronous:
            sync(self)
      segmap_usable_groups_64_16111 = sdiv_up64(nest_sizze_16004,
                                                segmap_group_sizze_16110)
      segmap_usable_groups_16112 = sext_i64_i32(segmap_usable_groups_64_16111)
      mem_16488 = opencl_alloc(self, bytes_16457, "mem_16488")
      if ((1 * (np.long(segmap_usable_groups_16112) * np.long(segmap_group_sizze_16109))) != 0):
        self.tridagNestedzisegmap_15803_var.set_args(self.global_failure,
                                                     np.int32(n_13843),
                                                     np.int32(m_13844),
                                                     np.int32(m_13846),
                                                     b_mem_16410, mem_16467,
                                                     mem_16472, mem_16477,
                                                     mem_16482, mem_16488)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedzisegmap_15803_var,
                                   ((np.long(segmap_usable_groups_16112) * np.long(segmap_group_sizze_16109)),),
                                   (np.long(segmap_group_sizze_16109),))
        if synchronous:
          sync(self)
      mem_16467 = None
      mem_16472 = None
      mem_16477 = None
      mem_16482 = None
      mem_16494 = opencl_alloc(self, bytes_16457, "mem_16494")
      mem_16499 = opencl_alloc(self, bytes_16457, "mem_16499")
      if slt32(np.int32(0), (n_13843 * m_13844)):
        stage1_max_num_groups_16815 = self.max_group_size
        stage1_num_groups_16816 = smin32(stage1_max_num_groups_16815,
                                         num_groups_16174)
        num_threads_16817 = (stage1_num_groups_16816 * segscan_group_sizze_16173)
        if ((1 * (np.long(stage1_num_groups_16816) * np.long(segscan_group_sizze_16173))) != 0):
          self.tridagNestedziscan_stage1_15697_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_16173))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_16173))))),
                                                            np.int32(n_13843),
                                                            np.int32(m_13844),
                                                            np.int32(m_13850),
                                                            a_mem_16409,
                                                            y_mem_16412,
                                                            mem_16488,
                                                            mem_16494,
                                                            mem_16499,
                                                            np.int32(num_threads_16817))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage1_15697_var,
                                     ((np.long(stage1_num_groups_16816) * np.long(segscan_group_sizze_16173)),),
                                     (np.long(segscan_group_sizze_16173),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_16816))) != 0):
          self.tridagNestedziscan_stage2_15697_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16816))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16816))))),
                                                            np.int32(n_13843),
                                                            np.int32(m_13844),
                                                            mem_16494,
                                                            mem_16499,
                                                            np.int32(stage1_num_groups_16816),
                                                            np.int32(num_threads_16817))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage2_15697_var,
                                     ((np.long(np.int32(1)) * np.long(stage1_num_groups_16816)),),
                                     (np.long(stage1_num_groups_16816),))
          if synchronous:
            sync(self)
        required_groups_16873 = sdiv_up32((n_13843 * m_13844),
                                          segscan_group_sizze_16173)
        if ((1 * (np.long(num_groups_16174) * np.long(segscan_group_sizze_16173))) != 0):
          self.tridagNestedziscan_stage3_15697_var.set_args(self.global_failure,
                                                            np.int32(n_13843),
                                                            np.int32(m_13844),
                                                            np.int32(num_groups_16174),
                                                            mem_16494,
                                                            mem_16499,
                                                            np.int32(num_threads_16817),
                                                            np.int32(required_groups_16873))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage3_15697_var,
                                     ((np.long(num_groups_16174) * np.long(segscan_group_sizze_16173)),),
                                     (np.long(segscan_group_sizze_16173),))
          if synchronous:
            sync(self)
      segmap_usable_groups_64_16235 = sdiv_up64(nest_sizze_16004,
                                                segmap_group_sizze_16234)
      segmap_usable_groups_16236 = sext_i64_i32(segmap_usable_groups_64_16235)
      mem_16505 = opencl_alloc(self, bytes_16457, "mem_16505")
      if ((1 * (np.long(segmap_usable_groups_16236) * np.long(segmap_group_sizze_16233))) != 0):
        self.tridagNestedzisegmap_15628_var.set_args(self.global_failure,
                                                     np.int32(n_13843),
                                                     np.int32(m_13844),
                                                     np.int32(m_13850),
                                                     y_mem_16412, mem_16494,
                                                     mem_16499, mem_16505)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedzisegmap_15628_var,
                                   ((np.long(segmap_usable_groups_16236) * np.long(segmap_group_sizze_16233)),),
                                   (np.long(segmap_group_sizze_16233),))
        if synchronous:
          sync(self)
      mem_16494 = None
      mem_16499 = None
      mem_16509 = opencl_alloc(self, bytes_16507, "mem_16509")
      if ((1 * (np.long(num_groups_16261) * np.long(segmap_group_sizze_16260))) != 0):
        self.tridagNestedzisegmap_15567_var.set_args(self.global_failure,
                                                     np.int32(n_13843),
                                                     np.int32(m_13844),
                                                     np.int32(i_13871),
                                                     np.int32(num_groups_16261),
                                                     mem_16488, mem_16505,
                                                     mem_16509)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedzisegmap_15567_var,
                                   ((np.long(num_groups_16261) * np.long(segmap_group_sizze_16260)),),
                                   (np.long(segmap_group_sizze_16260),))
        if synchronous:
          sync(self)
      mem_16515 = opencl_alloc(self, bytes_16457, "mem_16515")
      mem_16520 = opencl_alloc(self, bytes_16457, "mem_16520")
      if slt32(np.int32(0), (n_13843 * m_13844)):
        stage1_max_num_groups_16898 = self.max_group_size
        stage1_num_groups_16899 = smin32(stage1_max_num_groups_16898,
                                         num_groups_16279)
        num_threads_16900 = (stage1_num_groups_16899 * segscan_group_sizze_16278)
        if ((1 * (np.long(stage1_num_groups_16899) * np.long(segscan_group_sizze_16278))) != 0):
          self.tridagNestedziscan_stage1_15542_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_16278))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_16278))))),
                                                            np.int32(n_13843),
                                                            np.int32(m_13844),
                                                            np.int32(m_13848),
                                                            c_mem_16411,
                                                            mem_16488,
                                                            mem_16505,
                                                            mem_16515,
                                                            mem_16520,
                                                            np.int32(num_threads_16900))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage1_15542_var,
                                     ((np.long(stage1_num_groups_16899) * np.long(segscan_group_sizze_16278)),),
                                     (np.long(segscan_group_sizze_16278),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_16899))) != 0):
          self.tridagNestedziscan_stage2_15542_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16899))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16899))))),
                                                            np.int32(n_13843),
                                                            np.int32(m_13844),
                                                            mem_16515,
                                                            mem_16520,
                                                            np.int32(stage1_num_groups_16899),
                                                            np.int32(num_threads_16900))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage2_15542_var,
                                     ((np.long(np.int32(1)) * np.long(stage1_num_groups_16899)),),
                                     (np.long(stage1_num_groups_16899),))
          if synchronous:
            sync(self)
        required_groups_16956 = sdiv_up32((n_13843 * m_13844),
                                          segscan_group_sizze_16278)
        if ((1 * (np.long(num_groups_16279) * np.long(segscan_group_sizze_16278))) != 0):
          self.tridagNestedziscan_stage3_15542_var.set_args(self.global_failure,
                                                            np.int32(n_13843),
                                                            np.int32(m_13844),
                                                            np.int32(num_groups_16279),
                                                            mem_16515,
                                                            mem_16520,
                                                            np.int32(num_threads_16900),
                                                            np.int32(required_groups_16956))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage3_15542_var,
                                     ((np.long(num_groups_16279) * np.long(segscan_group_sizze_16278)),),
                                     (np.long(segscan_group_sizze_16278),))
          if synchronous:
            sync(self)
      mem_16488 = None
      mem_16505 = None
      segmap_usable_groups_64_16342 = sdiv_up64(nest_sizze_16004,
                                                segmap_group_sizze_16341)
      segmap_usable_groups_16343 = sext_i64_i32(segmap_usable_groups_64_16342)
      mem_16526 = opencl_alloc(self, bytes_16457, "mem_16526")
      if ((1 * (np.long(segmap_usable_groups_16343) * np.long(segmap_group_sizze_16340))) != 0):
        self.tridagNestedzisegmap_15473_var.set_args(self.global_failure,
                                                     np.int32(n_13843),
                                                     np.int32(m_13844),
                                                     mem_16509, mem_16515,
                                                     mem_16520, mem_16526)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedzisegmap_15473_var,
                                   ((np.long(segmap_usable_groups_16343) * np.long(segmap_group_sizze_16340)),),
                                   (np.long(segmap_group_sizze_16340),))
        if synchronous:
          sync(self)
      mem_16509 = None
      mem_16515 = None
      mem_16520 = None
      segmap_usable_groups_64_16382 = sdiv_up64(nest_sizze_16004,
                                                segmap_group_sizze_16381)
      segmap_usable_groups_16383 = sext_i64_i32(segmap_usable_groups_64_16382)
      mem_16532 = opencl_alloc(self, bytes_16457, "mem_16532")
      if ((1 * (np.long(segmap_usable_groups_16383) * np.long(segmap_group_sizze_16380))) != 0):
        self.tridagNestedzisegmap_15399_var.set_args(self.global_failure,
                                                     np.int32(n_13843),
                                                     np.int32(m_13844),
                                                     mem_16526, mem_16532)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedzisegmap_15399_var,
                                   ((np.long(segmap_usable_groups_16383) * np.long(segmap_group_sizze_16380)),),
                                   (np.long(segmap_group_sizze_16380),))
        if synchronous:
          sync(self)
      mem_16526 = None
      res_mem_16533 = mem_16532
    out_arrsizze_16601 = n_13843
    out_arrsizze_16602 = m_13844
    out_mem_16600 = res_mem_16533
    return (out_mem_16600, out_arrsizze_16601, out_arrsizze_16602)
  def futhark_tridagParFlat(self, a_mem_16409, b_mem_16410, c_mem_16411,
                            y_mem_16412, n_13426, n_13427, n_13428, n_13429,
                            segSizze_13434, segCount_13435):
    dim_match_13436 = (n_13426 == n_13427)
    empty_or_match_cert_13437 = True
    assert dim_match_13436, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:111:1-196:7\n" % ("function arguments of wrong shape",))
    dim_match_13438 = (n_13426 == n_13428)
    empty_or_match_cert_13439 = True
    assert dim_match_13438, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:111:1-196:7\n" % ("function arguments of wrong shape",))
    dim_match_13440 = (n_13426 == n_13429)
    empty_or_match_cert_13441 = True
    assert dim_match_13440, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:111:1-196:7\n" % ("function arguments of wrong shape",))
    binop_x_16414 = sext_i32_i64(n_13426)
    bytes_16413 = (np.int64(4) * binop_x_16414)
    mem_16415 = opencl_alloc(self, bytes_16413, "mem_16415")
    self.futhark_builtinzhreplicate_i32(mem_16415, n_13426, np.int32(0))
    bytes_16416 = (np.int64(8) * binop_x_16414)
    mem_16418 = opencl_alloc(self, bytes_16416, "mem_16418")
    self.futhark_builtinzhreplicate_f64(mem_16418, n_13426, np.float64(0.0))
    segCount_14131 = sext_i32_i64(segCount_13435)
    segmap_group_sizze_14134 = self.sizes["tridagParFlat.segmap_group_size_14133"]
    segmap_group_sizze_14135 = sext_i32_i64(segmap_group_sizze_14134)
    segmap_usable_groups_64_14136 = sdiv_up64(segCount_14131,
                                              segmap_group_sizze_14135)
    segmap_usable_groups_14137 = sext_i64_i32(segmap_usable_groups_64_14136)
    if ((1 * (np.long(segmap_usable_groups_14137) * np.long(segmap_group_sizze_14134))) != 0):
      self.tridagParFlatzisegmap_14130_var.set_args(self.global_failure,
                                                    np.int32(n_13426),
                                                    np.int32(segSizze_13434),
                                                    np.int32(segCount_13435),
                                                    b_mem_16410, mem_16415,
                                                    mem_16418)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.tridagParFlatzisegmap_14130_var,
                                 ((np.long(segmap_usable_groups_14137) * np.long(segmap_group_sizze_14134)),),
                                 (np.long(segmap_group_sizze_14134),))
      if synchronous:
        sync(self)
    segscan_group_sizze_14141 = self.sizes["tridagParFlat.segscan_group_size_14140"]
    max_num_groups_16625 = self.sizes["tridagParFlat.segscan_num_groups_14142"]
    num_groups_14143 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(binop_x_16414,
                                                            sext_i32_i64(segscan_group_sizze_14141)),
                                                  sext_i32_i64(max_num_groups_16625))))
    mem_16423 = opencl_alloc(self, bytes_16413, "mem_16423")
    mem_16426 = opencl_alloc(self, bytes_16416, "mem_16426")
    if slt32(np.int32(0), n_13426):
      stage1_max_num_groups_16626 = self.max_group_size
      stage1_num_groups_16627 = smin32(stage1_max_num_groups_16626,
                                       num_groups_14143)
      num_threads_16628 = (stage1_num_groups_16627 * segscan_group_sizze_14141)
      if ((1 * (np.long(stage1_num_groups_16627) * np.long(segscan_group_sizze_14141))) != 0):
        self.tridagParFlatziscan_stage1_14146_var.set_args(self.global_failure,
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_14141))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(4)) * sext_i32_i64(segscan_group_sizze_14141))))),
                                                           np.int32(n_13426),
                                                           mem_16418, mem_16423,
                                                           mem_16426,
                                                           np.int32(num_threads_16628))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage1_14146_var,
                                   ((np.long(stage1_num_groups_16627) * np.long(segscan_group_sizze_14141)),),
                                   (np.long(segscan_group_sizze_14141),))
        if synchronous:
          sync(self)
      if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_16627))) != 0):
        self.tridagParFlatziscan_stage2_14146_var.set_args(self.global_failure,
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16627))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(4)) * sext_i32_i64(stage1_num_groups_16627))))),
                                                           np.int32(n_13426),
                                                           mem_16423, mem_16426,
                                                           np.int32(stage1_num_groups_16627),
                                                           np.int32(num_threads_16628))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage2_14146_var,
                                   ((np.long(np.int32(1)) * np.long(stage1_num_groups_16627)),),
                                   (np.long(stage1_num_groups_16627),))
        if synchronous:
          sync(self)
      required_groups_16681 = sdiv_up32(n_13426, segscan_group_sizze_14141)
      if ((1 * (np.long(num_groups_14143) * np.long(segscan_group_sizze_14141))) != 0):
        self.tridagParFlatziscan_stage3_14146_var.set_args(self.global_failure,
                                                           np.int32(n_13426),
                                                           np.int32(num_groups_14143),
                                                           mem_16423, mem_16426,
                                                           np.int32(num_threads_16628),
                                                           np.int32(required_groups_16681))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage3_14146_var,
                                   ((np.long(num_groups_14143) * np.long(segscan_group_sizze_14141)),),
                                   (np.long(segscan_group_sizze_14141),))
        if synchronous:
          sync(self)
    mem_16418 = None
    mem_16423 = None
    segscan_group_sizze_14150 = self.sizes["tridagParFlat.segscan_group_size_14149"]
    max_num_groups_16693 = self.sizes["tridagParFlat.segscan_num_groups_14151"]
    num_groups_14152 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(binop_x_16414,
                                                            sext_i32_i64(segscan_group_sizze_14150)),
                                                  sext_i32_i64(max_num_groups_16693))))
    mem_16430 = opencl_alloc(self, bytes_16413, "mem_16430")
    mem_16433 = opencl_alloc(self, bytes_16416, "mem_16433")
    mem_16436 = opencl_alloc(self, bytes_16416, "mem_16436")
    mem_16439 = opencl_alloc(self, bytes_16416, "mem_16439")
    mem_16442 = opencl_alloc(self, bytes_16416, "mem_16442")
    if slt32(np.int32(0), n_13426):
      stage1_max_num_groups_16694 = self.max_group_size
      stage1_num_groups_16695 = smin32(stage1_max_num_groups_16694,
                                       num_groups_14152)
      num_threads_16696 = (stage1_num_groups_16695 * segscan_group_sizze_14150)
      if ((1 * (np.long(stage1_num_groups_16695) * np.long(segscan_group_sizze_14150))) != 0):
        self.tridagParFlatziscan_stage1_14155_var.set_args(self.global_failure,
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_14150))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_14150))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_14150))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_14150))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(4)) * sext_i32_i64(segscan_group_sizze_14150))))),
                                                           np.int32(n_13426),
                                                           np.int32(segSizze_13434),
                                                           a_mem_16409,
                                                           b_mem_16410,
                                                           c_mem_16411,
                                                           mem_16415, mem_16430,
                                                           mem_16433, mem_16436,
                                                           mem_16439, mem_16442,
                                                           np.int32(num_threads_16696))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage1_14155_var,
                                   ((np.long(stage1_num_groups_16695) * np.long(segscan_group_sizze_14150)),),
                                   (np.long(segscan_group_sizze_14150),))
        if synchronous:
          sync(self)
      if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_16695))) != 0):
        self.tridagParFlatziscan_stage2_14155_var.set_args(self.global_failure,
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16695))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16695))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16695))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16695))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(4)) * sext_i32_i64(stage1_num_groups_16695))))),
                                                           np.int32(n_13426),
                                                           mem_16430, mem_16433,
                                                           mem_16436, mem_16439,
                                                           mem_16442,
                                                           np.int32(stage1_num_groups_16695),
                                                           np.int32(num_threads_16696))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage2_14155_var,
                                   ((np.long(np.int32(1)) * np.long(stage1_num_groups_16695)),),
                                   (np.long(stage1_num_groups_16695),))
        if synchronous:
          sync(self)
      required_groups_16836 = sdiv_up32(n_13426, segscan_group_sizze_14150)
      if ((1 * (np.long(num_groups_14152) * np.long(segscan_group_sizze_14150))) != 0):
        self.tridagParFlatziscan_stage3_14155_var.set_args(self.global_failure,
                                                           np.int32(n_13426),
                                                           np.int32(num_groups_14152),
                                                           mem_16430, mem_16433,
                                                           mem_16436, mem_16439,
                                                           mem_16442,
                                                           np.int32(num_threads_16696),
                                                           np.int32(required_groups_16836))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage3_14155_var,
                                   ((np.long(num_groups_14152) * np.long(segscan_group_sizze_14150)),),
                                   (np.long(segscan_group_sizze_14150),))
        if synchronous:
          sync(self)
    segmap_group_sizze_14296 = self.sizes["tridagParFlat.segmap_group_size_14252"]
    segmap_group_sizze_14297 = sext_i32_i64(segmap_group_sizze_14296)
    segmap_usable_groups_64_14298 = sdiv_up64(binop_x_16414,
                                              segmap_group_sizze_14297)
    segmap_usable_groups_14299 = sext_i64_i32(segmap_usable_groups_64_14298)
    mem_16446 = opencl_alloc(self, bytes_16416, "mem_16446")
    if ((1 * (np.long(segmap_usable_groups_14299) * np.long(segmap_group_sizze_14296))) != 0):
      self.tridagParFlatzisegmap_14249_var.set_args(self.global_failure,
                                                    np.int32(n_13426),
                                                    mem_16426, mem_16430,
                                                    mem_16433, mem_16436,
                                                    mem_16439, mem_16442,
                                                    mem_16446)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.tridagParFlatzisegmap_14249_var,
                                 ((np.long(segmap_usable_groups_14299) * np.long(segmap_group_sizze_14296)),),
                                 (np.long(segmap_group_sizze_14296),))
      if synchronous:
        sync(self)
    mem_16426 = None
    mem_16430 = None
    mem_16433 = None
    mem_16436 = None
    mem_16439 = None
    mem_16442 = None
    mem_16449 = opencl_alloc(self, bytes_16416, "mem_16449")
    self.futhark_builtinzhreplicate_f64(mem_16449, n_13426, np.float64(0.0))
    segmap_group_sizze_14343 = self.sizes["tridagParFlat.segmap_group_size_14342"]
    segmap_group_sizze_14344 = sext_i32_i64(segmap_group_sizze_14343)
    segmap_usable_groups_64_14345 = sdiv_up64(segCount_14131,
                                              segmap_group_sizze_14344)
    segmap_usable_groups_14346 = sext_i64_i32(segmap_usable_groups_64_14345)
    if ((1 * (np.long(segmap_usable_groups_14346) * np.long(segmap_group_sizze_14343))) != 0):
      self.tridagParFlatzisegmap_14339_var.set_args(self.global_failure,
                                                    np.int32(n_13426),
                                                    np.int32(segSizze_13434),
                                                    np.int32(segCount_13435),
                                                    y_mem_16412, mem_16449)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.tridagParFlatzisegmap_14339_var,
                                 ((np.long(segmap_usable_groups_14346) * np.long(segmap_group_sizze_14343)),),
                                 (np.long(segmap_group_sizze_14343),))
      if synchronous:
        sync(self)
    segscan_group_sizze_14350 = self.sizes["tridagParFlat.segscan_group_size_14349"]
    max_num_groups_16858 = self.sizes["tridagParFlat.segscan_num_groups_14351"]
    num_groups_14352 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(binop_x_16414,
                                                            sext_i32_i64(segscan_group_sizze_14350)),
                                                  sext_i32_i64(max_num_groups_16858))))
    mem_16454 = opencl_alloc(self, bytes_16413, "mem_16454")
    mem_16457 = opencl_alloc(self, bytes_16416, "mem_16457")
    if slt32(np.int32(0), n_13426):
      stage1_max_num_groups_16859 = self.max_group_size
      stage1_num_groups_16860 = smin32(stage1_max_num_groups_16859,
                                       num_groups_14352)
      num_threads_16861 = (stage1_num_groups_16860 * segscan_group_sizze_14350)
      if ((1 * (np.long(stage1_num_groups_16860) * np.long(segscan_group_sizze_14350))) != 0):
        self.tridagParFlatziscan_stage1_14355_var.set_args(self.global_failure,
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_14350))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(4)) * sext_i32_i64(segscan_group_sizze_14350))))),
                                                           np.int32(n_13426),
                                                           mem_16449, mem_16454,
                                                           mem_16457,
                                                           np.int32(num_threads_16861))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage1_14355_var,
                                   ((np.long(stage1_num_groups_16860) * np.long(segscan_group_sizze_14350)),),
                                   (np.long(segscan_group_sizze_14350),))
        if synchronous:
          sync(self)
      if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_16860))) != 0):
        self.tridagParFlatziscan_stage2_14355_var.set_args(self.global_failure,
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16860))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(4)) * sext_i32_i64(stage1_num_groups_16860))))),
                                                           np.int32(n_13426),
                                                           mem_16454, mem_16457,
                                                           np.int32(stage1_num_groups_16860),
                                                           np.int32(num_threads_16861))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage2_14355_var,
                                   ((np.long(np.int32(1)) * np.long(stage1_num_groups_16860)),),
                                   (np.long(stage1_num_groups_16860),))
        if synchronous:
          sync(self)
      required_groups_16914 = sdiv_up32(n_13426, segscan_group_sizze_14350)
      if ((1 * (np.long(num_groups_14352) * np.long(segscan_group_sizze_14350))) != 0):
        self.tridagParFlatziscan_stage3_14355_var.set_args(self.global_failure,
                                                           np.int32(n_13426),
                                                           np.int32(num_groups_14352),
                                                           mem_16454, mem_16457,
                                                           np.int32(num_threads_16861),
                                                           np.int32(required_groups_16914))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage3_14355_var,
                                   ((np.long(num_groups_14352) * np.long(segscan_group_sizze_14350)),),
                                   (np.long(segscan_group_sizze_14350),))
        if synchronous:
          sync(self)
    mem_16449 = None
    mem_16454 = None
    segscan_group_sizze_14359 = self.sizes["tridagParFlat.segscan_group_size_14358"]
    max_num_groups_16926 = self.sizes["tridagParFlat.segscan_num_groups_14360"]
    num_groups_14361 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(binop_x_16414,
                                                            sext_i32_i64(segscan_group_sizze_14359)),
                                                  sext_i32_i64(max_num_groups_16926))))
    mem_16461 = opencl_alloc(self, bytes_16413, "mem_16461")
    mem_16464 = opencl_alloc(self, bytes_16416, "mem_16464")
    mem_16467 = opencl_alloc(self, bytes_16416, "mem_16467")
    if slt32(np.int32(0), n_13426):
      stage1_max_num_groups_16927 = self.max_group_size
      stage1_num_groups_16928 = smin32(stage1_max_num_groups_16927,
                                       num_groups_14361)
      num_threads_16929 = (stage1_num_groups_16928 * segscan_group_sizze_14359)
      if ((1 * (np.long(stage1_num_groups_16928) * np.long(segscan_group_sizze_14359))) != 0):
        self.tridagParFlatziscan_stage1_14364_var.set_args(self.global_failure,
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_14359))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_14359))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(4)) * sext_i32_i64(segscan_group_sizze_14359))))),
                                                           np.int32(n_13426),
                                                           np.int32(segSizze_13434),
                                                           a_mem_16409,
                                                           y_mem_16412,
                                                           mem_16415, mem_16446,
                                                           mem_16461, mem_16464,
                                                           mem_16467,
                                                           np.int32(num_threads_16929))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage1_14364_var,
                                   ((np.long(stage1_num_groups_16928) * np.long(segscan_group_sizze_14359)),),
                                   (np.long(segscan_group_sizze_14359),))
        if synchronous:
          sync(self)
      if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_16928))) != 0):
        self.tridagParFlatziscan_stage2_14364_var.set_args(self.global_failure,
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16928))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16928))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(4)) * sext_i32_i64(stage1_num_groups_16928))))),
                                                           np.int32(n_13426),
                                                           mem_16461, mem_16464,
                                                           mem_16467,
                                                           np.int32(stage1_num_groups_16928),
                                                           np.int32(num_threads_16929))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage2_14364_var,
                                   ((np.long(np.int32(1)) * np.long(stage1_num_groups_16928)),),
                                   (np.long(stage1_num_groups_16928),))
        if synchronous:
          sync(self)
      required_groups_17001 = sdiv_up32(n_13426, segscan_group_sizze_14359)
      if ((1 * (np.long(num_groups_14361) * np.long(segscan_group_sizze_14359))) != 0):
        self.tridagParFlatziscan_stage3_14364_var.set_args(self.global_failure,
                                                           np.int32(n_13426),
                                                           np.int32(num_groups_14361),
                                                           mem_16461, mem_16464,
                                                           mem_16467,
                                                           np.int32(num_threads_16929),
                                                           np.int32(required_groups_17001))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage3_14364_var,
                                   ((np.long(num_groups_14361) * np.long(segscan_group_sizze_14359)),),
                                   (np.long(segscan_group_sizze_14359),))
        if synchronous:
          sync(self)
    mem_16461 = None
    segmap_group_sizze_14442 = self.sizes["tridagParFlat.segmap_group_size_14419"]
    segmap_group_sizze_14443 = sext_i32_i64(segmap_group_sizze_14442)
    segmap_usable_groups_64_14444 = sdiv_up64(binop_x_16414,
                                              segmap_group_sizze_14443)
    segmap_usable_groups_14445 = sext_i64_i32(segmap_usable_groups_64_14444)
    mem_16471 = opencl_alloc(self, bytes_16416, "mem_16471")
    if ((1 * (np.long(segmap_usable_groups_14445) * np.long(segmap_group_sizze_14442))) != 0):
      self.tridagParFlatzisegmap_14416_var.set_args(self.global_failure,
                                                    np.int32(n_13426),
                                                    mem_16457, mem_16464,
                                                    mem_16467, mem_16471)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.tridagParFlatzisegmap_14416_var,
                                 ((np.long(segmap_usable_groups_14445) * np.long(segmap_group_sizze_14442)),),
                                 (np.long(segmap_group_sizze_14442),))
      if synchronous:
        sync(self)
    mem_16457 = None
    mem_16464 = None
    mem_16467 = None
    mem_16474 = opencl_alloc(self, bytes_16416, "mem_16474")
    self.futhark_builtinzhreplicate_f64(mem_16474, n_13426, np.float64(0.0))
    segmap_group_sizze_14468 = self.sizes["tridagParFlat.segmap_group_size_14467"]
    segmap_group_sizze_14469 = sext_i32_i64(segmap_group_sizze_14468)
    segmap_usable_groups_64_14470 = sdiv_up64(segCount_14131,
                                              segmap_group_sizze_14469)
    segmap_usable_groups_14471 = sext_i64_i32(segmap_usable_groups_64_14470)
    if ((1 * (np.long(segmap_usable_groups_14471) * np.long(segmap_group_sizze_14468))) != 0):
      self.tridagParFlatzisegmap_14464_var.set_args(self.global_failure,
                                                    np.int32(n_13426),
                                                    np.int32(segSizze_13434),
                                                    np.int32(segCount_13435),
                                                    mem_16446, mem_16471,
                                                    mem_16474)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.tridagParFlatzisegmap_14464_var,
                                 ((np.long(segmap_usable_groups_14471) * np.long(segmap_group_sizze_14468)),),
                                 (np.long(segmap_group_sizze_14468),))
      if synchronous:
        sync(self)
    segscan_group_sizze_14475 = self.sizes["tridagParFlat.segscan_group_size_14474"]
    max_num_groups_17023 = self.sizes["tridagParFlat.segscan_num_groups_14476"]
    num_groups_14477 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(binop_x_16414,
                                                            sext_i32_i64(segscan_group_sizze_14475)),
                                                  sext_i32_i64(max_num_groups_17023))))
    mem_16479 = opencl_alloc(self, bytes_16413, "mem_16479")
    mem_16482 = opencl_alloc(self, bytes_16416, "mem_16482")
    if slt32(np.int32(0), n_13426):
      stage1_max_num_groups_17024 = self.max_group_size
      stage1_num_groups_17025 = smin32(stage1_max_num_groups_17024,
                                       num_groups_14477)
      num_threads_17026 = (stage1_num_groups_17025 * segscan_group_sizze_14475)
      if ((1 * (np.long(stage1_num_groups_17025) * np.long(segscan_group_sizze_14475))) != 0):
        self.tridagParFlatziscan_stage1_14480_var.set_args(self.global_failure,
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_14475))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(4)) * sext_i32_i64(segscan_group_sizze_14475))))),
                                                           np.int32(n_13426),
                                                           mem_16474, mem_16479,
                                                           mem_16482,
                                                           np.int32(num_threads_17026))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage1_14480_var,
                                   ((np.long(stage1_num_groups_17025) * np.long(segscan_group_sizze_14475)),),
                                   (np.long(segscan_group_sizze_14475),))
        if synchronous:
          sync(self)
      if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_17025))) != 0):
        self.tridagParFlatziscan_stage2_14480_var.set_args(self.global_failure,
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_17025))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(4)) * sext_i32_i64(stage1_num_groups_17025))))),
                                                           np.int32(n_13426),
                                                           mem_16479, mem_16482,
                                                           np.int32(stage1_num_groups_17025),
                                                           np.int32(num_threads_17026))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage2_14480_var,
                                   ((np.long(np.int32(1)) * np.long(stage1_num_groups_17025)),),
                                   (np.long(stage1_num_groups_17025),))
        if synchronous:
          sync(self)
      required_groups_17079 = sdiv_up32(n_13426, segscan_group_sizze_14475)
      if ((1 * (np.long(num_groups_14477) * np.long(segscan_group_sizze_14475))) != 0):
        self.tridagParFlatziscan_stage3_14480_var.set_args(self.global_failure,
                                                           np.int32(n_13426),
                                                           np.int32(num_groups_14477),
                                                           mem_16479, mem_16482,
                                                           np.int32(num_threads_17026),
                                                           np.int32(required_groups_17079))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage3_14480_var,
                                   ((np.long(num_groups_14477) * np.long(segscan_group_sizze_14475)),),
                                   (np.long(segscan_group_sizze_14475),))
        if synchronous:
          sync(self)
    mem_16474 = None
    mem_16479 = None
    segscan_group_sizze_14484 = self.sizes["tridagParFlat.segscan_group_size_14483"]
    max_num_groups_17091 = self.sizes["tridagParFlat.segscan_num_groups_14485"]
    num_groups_14486 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(binop_x_16414,
                                                            sext_i32_i64(segscan_group_sizze_14484)),
                                                  sext_i32_i64(max_num_groups_17091))))
    mem_16486 = opencl_alloc(self, bytes_16413, "mem_16486")
    mem_16489 = opencl_alloc(self, bytes_16416, "mem_16489")
    mem_16492 = opencl_alloc(self, bytes_16416, "mem_16492")
    if slt32(np.int32(0), n_13426):
      stage1_max_num_groups_17092 = self.max_group_size
      stage1_num_groups_17093 = smin32(stage1_max_num_groups_17092,
                                       num_groups_14486)
      num_threads_17094 = (stage1_num_groups_17093 * segscan_group_sizze_14484)
      if ((1 * (np.long(stage1_num_groups_17093) * np.long(segscan_group_sizze_14484))) != 0):
        self.tridagParFlatziscan_stage1_14489_var.set_args(self.global_failure,
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_14484))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_14484))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(4)) * sext_i32_i64(segscan_group_sizze_14484))))),
                                                           np.int32(n_13426),
                                                           np.int32(segSizze_13434),
                                                           c_mem_16411,
                                                           mem_16415, mem_16446,
                                                           mem_16471, mem_16486,
                                                           mem_16489, mem_16492,
                                                           np.int32(num_threads_17094))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage1_14489_var,
                                   ((np.long(stage1_num_groups_17093) * np.long(segscan_group_sizze_14484)),),
                                   (np.long(segscan_group_sizze_14484),))
        if synchronous:
          sync(self)
      if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_17093))) != 0):
        self.tridagParFlatziscan_stage2_14489_var.set_args(self.global_failure,
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_17093))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_17093))))),
                                                           cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                         (sext_i32_i64(np.int32(4)) * sext_i32_i64(stage1_num_groups_17093))))),
                                                           np.int32(n_13426),
                                                           mem_16486, mem_16489,
                                                           mem_16492,
                                                           np.int32(stage1_num_groups_17093),
                                                           np.int32(num_threads_17094))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage2_14489_var,
                                   ((np.long(np.int32(1)) * np.long(stage1_num_groups_17093)),),
                                   (np.long(stage1_num_groups_17093),))
        if synchronous:
          sync(self)
      required_groups_17166 = sdiv_up32(n_13426, segscan_group_sizze_14484)
      if ((1 * (np.long(num_groups_14486) * np.long(segscan_group_sizze_14484))) != 0):
        self.tridagParFlatziscan_stage3_14489_var.set_args(self.global_failure,
                                                           np.int32(n_13426),
                                                           np.int32(num_groups_14486),
                                                           mem_16486, mem_16489,
                                                           mem_16492,
                                                           np.int32(num_threads_17094),
                                                           np.int32(required_groups_17166))
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagParFlatziscan_stage3_14489_var,
                                   ((np.long(num_groups_14486) * np.long(segscan_group_sizze_14484)),),
                                   (np.long(segscan_group_sizze_14484),))
        if synchronous:
          sync(self)
    mem_16415 = None
    mem_16446 = None
    mem_16471 = None
    mem_16486 = None
    segmap_group_sizze_14567 = self.sizes["tridagParFlat.segmap_group_size_14544"]
    segmap_group_sizze_14568 = sext_i32_i64(segmap_group_sizze_14567)
    segmap_usable_groups_64_14569 = sdiv_up64(binop_x_16414,
                                              segmap_group_sizze_14568)
    segmap_usable_groups_14570 = sext_i64_i32(segmap_usable_groups_64_14569)
    mem_16496 = opencl_alloc(self, bytes_16416, "mem_16496")
    if ((1 * (np.long(segmap_usable_groups_14570) * np.long(segmap_group_sizze_14567))) != 0):
      self.tridagParFlatzisegmap_14541_var.set_args(self.global_failure,
                                                    np.int32(n_13426),
                                                    mem_16482, mem_16489,
                                                    mem_16492, mem_16496)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.tridagParFlatzisegmap_14541_var,
                                 ((np.long(segmap_usable_groups_14570) * np.long(segmap_group_sizze_14567)),),
                                 (np.long(segmap_group_sizze_14567),))
      if synchronous:
        sync(self)
    mem_16482 = None
    mem_16489 = None
    mem_16492 = None
    segmap_group_sizze_14641 = self.sizes["tridagParFlat.segmap_group_size_14626"]
    segmap_group_sizze_14642 = sext_i32_i64(segmap_group_sizze_14641)
    segmap_usable_groups_64_14643 = sdiv_up64(binop_x_16414,
                                              segmap_group_sizze_14642)
    segmap_usable_groups_14644 = sext_i64_i32(segmap_usable_groups_64_14643)
    mem_16500 = opencl_alloc(self, bytes_16416, "mem_16500")
    if ((1 * (np.long(segmap_usable_groups_14644) * np.long(segmap_group_sizze_14641))) != 0):
      self.tridagParFlatzisegmap_14623_var.set_args(self.global_failure,
                                                    np.int32(n_13426),
                                                    np.int32(segSizze_13434),
                                                    mem_16496, mem_16500)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.tridagParFlatzisegmap_14623_var,
                                 ((np.long(segmap_usable_groups_14644) * np.long(segmap_group_sizze_14641)),),
                                 (np.long(segmap_group_sizze_14641),))
      if synchronous:
        sync(self)
    mem_16496 = None
    out_arrsizze_16601 = n_13426
    out_mem_16600 = mem_16500
    return (out_mem_16600, out_arrsizze_16601)
  def tridagNested(self, a_mem_16409_ext, b_mem_16410_ext, c_mem_16411_ext,
                   y_mem_16412_ext):
    try:
      assert ((type(a_mem_16409_ext) in [np.ndarray,
                                         cl.array.Array]) and (a_mem_16409_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_13843 = np.int32(a_mem_16409_ext.shape[0])
      m_13844 = np.int32(a_mem_16409_ext.shape[1])
      if (type(a_mem_16409_ext) == cl.array.Array):
        a_mem_16409 = a_mem_16409_ext.data
      else:
        a_mem_16409 = opencl_alloc(self, np.int64(a_mem_16409_ext.nbytes),
                                   "a_mem_16409")
        if (np.int64(a_mem_16409_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, a_mem_16409,
                          normaliseArray(a_mem_16409_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(a_mem_16409_ext),
                                                                                                                            a_mem_16409_ext))
    try:
      assert ((type(b_mem_16410_ext) in [np.ndarray,
                                         cl.array.Array]) and (b_mem_16410_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_13845 = np.int32(b_mem_16410_ext.shape[0])
      m_13846 = np.int32(b_mem_16410_ext.shape[1])
      if (type(b_mem_16410_ext) == cl.array.Array):
        b_mem_16410 = b_mem_16410_ext.data
      else:
        b_mem_16410 = opencl_alloc(self, np.int64(b_mem_16410_ext.nbytes),
                                   "b_mem_16410")
        if (np.int64(b_mem_16410_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, b_mem_16410,
                          normaliseArray(b_mem_16410_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(b_mem_16410_ext),
                                                                                                                            b_mem_16410_ext))
    try:
      assert ((type(c_mem_16411_ext) in [np.ndarray,
                                         cl.array.Array]) and (c_mem_16411_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_13847 = np.int32(c_mem_16411_ext.shape[0])
      m_13848 = np.int32(c_mem_16411_ext.shape[1])
      if (type(c_mem_16411_ext) == cl.array.Array):
        c_mem_16411 = c_mem_16411_ext.data
      else:
        c_mem_16411 = opencl_alloc(self, np.int64(c_mem_16411_ext.nbytes),
                                   "c_mem_16411")
        if (np.int64(c_mem_16411_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, c_mem_16411,
                          normaliseArray(c_mem_16411_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(c_mem_16411_ext),
                                                                                                                            c_mem_16411_ext))
    try:
      assert ((type(y_mem_16412_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_16412_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_13849 = np.int32(y_mem_16412_ext.shape[0])
      m_13850 = np.int32(y_mem_16412_ext.shape[1])
      if (type(y_mem_16412_ext) == cl.array.Array):
        y_mem_16412 = y_mem_16412_ext.data
      else:
        y_mem_16412 = opencl_alloc(self, np.int64(y_mem_16412_ext.nbytes),
                                   "y_mem_16412")
        if (np.int64(y_mem_16412_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_16412,
                          normaliseArray(y_mem_16412_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(y_mem_16412_ext),
                                                                                                                            y_mem_16412_ext))
    (out_mem_16600, out_arrsizze_16601,
     out_arrsizze_16602) = self.futhark_tridagNested(a_mem_16409, b_mem_16410,
                                                     c_mem_16411, y_mem_16412,
                                                     n_13843, m_13844, n_13845,
                                                     m_13846, n_13847, m_13848,
                                                     n_13849, m_13850)
    sync(self)
    return cl.array.Array(self.queue, (out_arrsizze_16601, out_arrsizze_16602),
                          ct.c_double, data=out_mem_16600)
  def tridagParFlat(self, a_mem_16409_ext, b_mem_16410_ext, c_mem_16411_ext,
                    y_mem_16412_ext, segSizze_13434_ext, segCount_13435_ext):
    try:
      assert ((type(a_mem_16409_ext) in [np.ndarray,
                                         cl.array.Array]) and (a_mem_16409_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_13426 = np.int32(a_mem_16409_ext.shape[0])
      if (type(a_mem_16409_ext) == cl.array.Array):
        a_mem_16409 = a_mem_16409_ext.data
      else:
        a_mem_16409 = opencl_alloc(self, np.int64(a_mem_16409_ext.nbytes),
                                   "a_mem_16409")
        if (np.int64(a_mem_16409_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, a_mem_16409,
                          normaliseArray(a_mem_16409_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f64",
                                                                                                                            type(a_mem_16409_ext),
                                                                                                                            a_mem_16409_ext))
    try:
      assert ((type(b_mem_16410_ext) in [np.ndarray,
                                         cl.array.Array]) and (b_mem_16410_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_13427 = np.int32(b_mem_16410_ext.shape[0])
      if (type(b_mem_16410_ext) == cl.array.Array):
        b_mem_16410 = b_mem_16410_ext.data
      else:
        b_mem_16410 = opencl_alloc(self, np.int64(b_mem_16410_ext.nbytes),
                                   "b_mem_16410")
        if (np.int64(b_mem_16410_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, b_mem_16410,
                          normaliseArray(b_mem_16410_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f64",
                                                                                                                            type(b_mem_16410_ext),
                                                                                                                            b_mem_16410_ext))
    try:
      assert ((type(c_mem_16411_ext) in [np.ndarray,
                                         cl.array.Array]) and (c_mem_16411_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_13428 = np.int32(c_mem_16411_ext.shape[0])
      if (type(c_mem_16411_ext) == cl.array.Array):
        c_mem_16411 = c_mem_16411_ext.data
      else:
        c_mem_16411 = opencl_alloc(self, np.int64(c_mem_16411_ext.nbytes),
                                   "c_mem_16411")
        if (np.int64(c_mem_16411_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, c_mem_16411,
                          normaliseArray(c_mem_16411_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f64",
                                                                                                                            type(c_mem_16411_ext),
                                                                                                                            c_mem_16411_ext))
    try:
      assert ((type(y_mem_16412_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_16412_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_13429 = np.int32(y_mem_16412_ext.shape[0])
      if (type(y_mem_16412_ext) == cl.array.Array):
        y_mem_16412 = y_mem_16412_ext.data
      else:
        y_mem_16412 = opencl_alloc(self, np.int64(y_mem_16412_ext.nbytes),
                                   "y_mem_16412")
        if (np.int64(y_mem_16412_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_16412,
                          normaliseArray(y_mem_16412_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f64",
                                                                                                                            type(y_mem_16412_ext),
                                                                                                                            y_mem_16412_ext))
    try:
      segSizze_13434 = np.int32(ct.c_int32(segSizze_13434_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(segSizze_13434_ext),
                                                                                                                            segSizze_13434_ext))
    try:
      segCount_13435 = np.int32(ct.c_int32(segCount_13435_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(segCount_13435_ext),
                                                                                                                            segCount_13435_ext))
    (out_mem_16600,
     out_arrsizze_16601) = self.futhark_tridagParFlat(a_mem_16409, b_mem_16410,
                                                      c_mem_16411, y_mem_16412,
                                                      n_13426, n_13427, n_13428,
                                                      n_13429, segSizze_13434,
                                                      segCount_13435)
    sync(self)
    return cl.array.Array(self.queue, (out_arrsizze_16601,), ct.c_double,
                          data=out_mem_16600)