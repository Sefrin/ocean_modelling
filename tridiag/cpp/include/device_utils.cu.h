#pragma once
#include <cuda_runtime.h>
template <typename T>
struct tuple4 
{
    T a = 1;
    T b = 0;
    T c = 0;
    T d = 1;
};
template <typename T>
struct tuple2
{
    T a = 0;
    T b = 1;
};

template <typename T>
class pbbtuple4 {
  public:
    T a; T b; T c; T d;

    __device__ __host__ inline pbbtuple4() {
        a = 1; b = 0; c = 0; d = 1; 
    }

    __device__ __host__ inline pbbtuple4(const T a, const T b, const T c, const T d) {
        this->a = a; this->b = b; this->c = c; this->d = d; 
    }

    __device__ __host__ inline pbbtuple4(const pbbtuple4<T>& t4) { 
        a = t4.a; b = t4.b; c = t4.c; d = t4.d; 
    }

    __device__ __host__ inline void operator=(const pbbtuple4<T>& t4) volatile{
        a = t4.a; b = t4.b; c = t4.c; d = t4.d; 
    }
};

template <typename T>
class pbbtuple2 {
  public:
    T a; T b;

    __device__ __host__ inline pbbtuple2() {
        a = 0; b = 1;
    }

    __device__ __host__ inline pbbtuple2(const T& a, const T& b) {
        a = a; b = b;
    }

    __device__ __host__ inline pbbtuple2(const pbbtuple2<T>& t1) { 
        a = t1.a; b = t1.b;
    }

    __device__ __host__ inline void operator=(const pbbtuple2<T>& t1) volatile{
        a = t1.a; b = t1.b;
    }
};

template<typename T>
class tuple2op {
  public:
    typedef pbbtuple2<T> InpElTp;
    typedef pbbtuple2<T> RedElTp;
   
    static __device__ __host__ inline pbbtuple2<T> apply(volatile pbbtuple2<T>& a, volatile pbbtuple2<T>& b) 
    {   
        pbbtuple2<T> res;
        res.a = b.a + b.b*a.a;
        res.b = a.b*b.b;
        return res;
    }

   
    static __device__ __host__ inline pbbtuple2<T> remVolatile(volatile pbbtuple2<T>& t)
    {
        pbbtuple2<T> res;
        res.a = t.a;
        res.b = t.b;
        return res;
    }
};

template<typename T>
class tuple4op {
  public:
    typedef pbbtuple4<T> InpElTp;
    typedef pbbtuple4<T> RedElTp;
   
    static __device__ __host__ inline pbbtuple4<T> apply(volatile pbbtuple4<T>& a, volatile pbbtuple4<T>& b) 
    {   
        T tmp = 1.0/(a.a*b.a);
        return pbbtuple4<T>(
            (b.a*a.a + b.b*a.c)*tmp,
            (b.a*a.b + b.b*a.d)*tmp,
            (b.c*a.a + b.d*a.c)*tmp,
            (b.c*a.b + b.d*a.d)*tmp
        );
    }

   
    static __device__ __host__ inline pbbtuple4<T> remVolatile(volatile pbbtuple4<T>& t)
    {
        pbbtuple4<T> res;
        res.a = t.a;
        res.b = t.b;
        res.c = t.c;
        res.d = t.d;
        return res;
    }
};


