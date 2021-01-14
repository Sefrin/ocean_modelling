import matplotlib.pyplot as plt

def plot_timings(x, size, prim_parallel, prim_parallel_const, prim_parallel_coal, prim_parallel_coal_const, flat, flat_shared, futhark_flat, futhark_seq):
    plt.xlabel("Number of systems")
    plt.ylabel("execution time (ms)")
    plt.title("Benchmarks of algorithms with systems of size " + str(size))
    plt.plot(x, prim_parallel, label = "Naive Thomas")
    plt.plot(x, prim_parallel_const, label = "Naive Thomas const size")
    plt.plot(x, prim_parallel_coal, label = "Thomas coalesced")
    plt.plot(x, prim_parallel_coal_const, label = "Thomas coalesced const size")
    plt.plot(x, flat, label = "Flattened algorithm")
    if flat_shared is not None:
        plt.plot(x, flat_shared, label = "Flattened algorithm intra block")
    if futhark_flat is not None:
        plt.plot(x, futhark_flat, label = "Flattened algorithm - Futhark")
        plt.plot(x, futhark_seq, label = "Thomas - Futhark")
    plt.legend(loc="upper left")
    plt.savefig("timings_" + str(size) +".png")
    plt.show()

##### Inner dim 15 
x = [1000, 10000, 50000, 100000]
prim_parallel = [0.04,0.11,1.41,2.87]
prim_parallel_const = [0.02,0.08,0.48,0.95]
prim_parallel_coal = [0.04,0.09,0.38,0.65]
prim_parallel_coal_const = [0.04,0.08,0.31,0.51]
flat = [0.16,0.42,1.48,2.47]
flat_shared = [0.09,0.78,3.42,6.33]
futhark_flat = [0.08,0.65,3.18,6.29]
futhark_seq = [0.10,0.13,0.47,0.84]

plot_timings(x, 15, prim_parallel, prim_parallel_const, prim_parallel_coal, prim_parallel_coal_const, flat, flat_shared, futhark_flat, futhark_seq)



##### Inner dim 115 
x = [1000, 10000, 50000, 100000]
prim_parallel = [0.29,2.41,24.21,50.01]
prim_parallel_const = [0.16,1.37,8.94,18.16]
prim_parallel_coal = [0.21,0.62,2.80,5.13]
prim_parallel_coal_const = [0.26,0.55,2.93,5.15]
flat = [0.37,2.23,9.80,19.43]
flat_shared = [0.24,2.09,9.48,18.68]
futhark_flat = [0.26,2.50,12.41,24.72]
futhark_seq = [0.29,0.76,3.65,6.76]

plot_timings(x, 115, prim_parallel, prim_parallel_const, prim_parallel_coal, prim_parallel_coal_const, flat, flat_shared, futhark_flat, futhark_seq)

##### Inner dim 1000  -- out of memory for 100000
x = [1000, 10000, 50000]
prim_parallel = [2.65,22.27,214.79]
prim_parallel_const = [1.43,11.02,78.93]
prim_parallel_coal = [1.97,5.15,25.41]
prim_parallel_coal_const = [1.50,5.06,25.20]
flat = [2.14,19.25,95.85]
flat_shared = [1.67,15.90,79.22]
futhark_flat = [2.15,20.35,100.01]
futhark_seq = [2.36,6.29,31.46]

plot_timings(x, 1000, prim_parallel, prim_parallel_const, prim_parallel_coal, prim_parallel_coal_const, flat, flat_shared, futhark_flat, futhark_seq)


##### Inner dim 10000  
x = [1, 10, 100]
prim_parallel = [5.59,6.48,13.62]
prim_parallel_const = [4.48,4.71,7.31]
prim_parallel_coal = [5.22,10.58,13.71]
prim_parallel_coal_const = [5.81,9.47,10.33]
flat = [0.15,0.33,2.27]
flat_shared = None
futhark_flat = None
futhark_seq = None

plot_timings(x, 10000, prim_parallel, prim_parallel_const, prim_parallel_coal, prim_parallel_coal_const, flat, flat_shared, futhark_flat, futhark_seq)
