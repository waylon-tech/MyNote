/*
 * Copyright (c) 1996-1997
 * Silicon Graphics Computer Systems, Inc.
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Silicon Graphics makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 */

// NOTE: This is an internal header file, included by other STL headers.
//       You should not attempt to use it directly.
//
// 注意：<stl_alloc.h> 被包含到其它的 STL 头文件中，并不是直接使用

#ifndef __SGI_STL_INTERNAL_ALLOC_H
#define __SGI_STL_INTERNAL_ALLOC_H

#ifdef __SUNPRO_CC
#define __PRIVATE public
// Extra access restrictions prevent us from really making some things
// private.
#else
#define __PRIVATE private
#endif

#ifdef __STL_STATIC_TEMPLATE_MEMBER_BUG
#define __USE_MALLOC
#endif

// This implements some standard node allocators.  These are
// NOT the same as the allocators in the C++ draft standard or in
// in the original STL.  They do not encapsulate different pointer
// types; indeed we assume that there is only one pointer type.
// The allocation primitives are intended to allocate individual objects,
// not larger arenas as with the original STL allocators.

// ==================================================
// 内存分配异常定义

#ifndef __THROW_BAD_ALLOC
#if defined(__STL_NO_BAD_ALLOC) || !defined(__STL_USE_EXCEPTIONS)
// 无 STL 异常，指针基于 C 的异常处理
#include <stdio.h>
#include <stdlib.h>
#define __THROW_BAD_ALLOC               \
    fprintf(stderr, "out of memory\n"); \
    exit(1)
#else /* Standard conforming out-of-memory handling */
// 有 STL 异常，制作基于 STL 的异常处理
#include <new>
#define __THROW_BAD_ALLOC throw std::bad_alloc()
#endif
#endif

// ==================================================
// 线程锁操作

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#ifndef __RESTRICT
#define __RESTRICT
#endif

#ifdef __STL_THREADS
// ----------------------------------------
// STL 线程锁机制
#include <stl_threads.h>
#define __NODE_ALLOCATOR_THREADS true
#ifdef __STL_SGI_THREADS
// We test whether threads are in use before locking.
// Perhaps this should be moved into stl_threads.h, but that
// probably makes it harder to avoid the procedure call when
// it isn't needed.
extern "C" // 向 C 兼容
{
    extern int __us_rsthread_malloc;
}
// The above is copied from malloc.h.  Including <malloc.h>
// would be cleaner but fails with certain levels of standard
// conformance.
#define __NODE_ALLOCATOR_LOCK                     \
    if (threads && __us_rsthread_malloc)          \
    {                                             \
        _S_node_allocator_lock._M_acquire_lock(); \
    }
#define __NODE_ALLOCATOR_UNLOCK                   \
    if (threads && __us_rsthread_malloc)          \
    {                                             \
        _S_node_allocator_lock._M_release_lock(); \
    }
#else /* !__STL_SGI_THREADS */
#define __NODE_ALLOCATOR_LOCK                         \
    {                                                 \
        if (threads)                                  \
            _S_node_allocator_lock._M_acquire_lock(); \
    }
#define __NODE_ALLOCATOR_UNLOCK                       \
    {                                                 \
        if (threads)                                  \
            _S_node_allocator_lock._M_release_lock(); \
    }
#endif
#else
// ----------------------------------------
// 自定义线程锁机制——无，因此是 Thread-unsafe 的
#define __NODE_ALLOCATOR_LOCK
#define __NODE_ALLOCATOR_UNLOCK
#define __NODE_ALLOCATOR_THREADS false
#endif

// ==================================================
// SGI STL 分配器

// ----------------------------------------
// out-of-memory 处理

__STL_BEGIN_NAMESPACE // 命名空间 宏：namespace std {

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1174 // 抑制名为 1374 的编译器警告
#endif

// Malloc-based allocator.  Typically slower than default alloc below.
// Typically thread-safe and more storage efficient.
#ifdef __STL_STATIC_TEMPLATE_MEMBER_BUG
#ifdef __DECLARE_GLOBALS_HERE
    // 定义全局变量于此
    void (*__malloc_alloc_oom_handler)() = 0;
// g++ 2.7.2 does not handle static template data members.
#else
    // 声明全局外部变量
    extern void (*__malloc_alloc_oom_handler)();
#endif
#endif

// ----------------------------------------
// SGI STL 第一级分配器

// SGI STL 第一级分配器，直接基于 malloc 和 free 来分配内存，用于大于 128byte 的内存管理。
// 无 “template 类型参数”，而 “非类型参数 __inst” 用于支持多个模板实例的创建（但也使容器的 allocator 类型不一致）
template <int __inst> 
class __malloc_alloc_template
{

private:
    // 以下是函数指针，所代表的函数用来处理内存不足的情况，在类外定义
    static void *_S_oom_malloc(size_t);
    static void *_S_oom_realloc(void *, size_t);
#ifndef __STL_STATIC_TEMPLATE_MEMBER_BUG
    static void (*__malloc_alloc_oom_handler)(); // out-of-memory 处理函数，在类外定义
#endif

public:
    // 分配函数
    static void *allocate(size_t __n)
    {
        void *__result = malloc(__n); // 第一级配置器直接调用 malloc()
        // 以下无法满足需求时，改用 _S_oom_malloc()
        if (0 == __result)
            __result = _S_oom_malloc(__n); // 分配失败，内存不足处理
        return __result;
    }

    // 销毁函数
    static void deallocate(void *__p, size_t /* __n */)
    {
        free(__p);  // 第一级分配器直接使用 free()
    }

    // 再分配函数
    static void *reallocate(void *__p, size_t /* old_sz */, size_t __new_sz)
    {
        void *__result = realloc(__p, __new_sz); // 第一级配置器直接调用 realloc()
        // 以下无法满足需求时，改用 _S_oom_realloc()
        if (0 == __result)
            __result = _S_oom_realloc(__p, __new_sz); // 分配失败，内存不足处理
        return __result;
    }

    // 自定义 out-of-memory 处理函数。
    // 第一级分配器会模仿 C++ 的 set_new_handler()，指定自己的 out-of-memory handler；
    // 这里不使用 C++ new-handler 机制，因为第一级配置器并没有用 ::operator new 来配置内存；
    // 【注1】C++ new handler 机制：要求系统 new 分配内存不成功时，调用一个称为 new handler 的自定义函数。
    // 		 new handler 处理内存不足的做法有特点的模式，参看《effective C++》2-e 条款 7.
    // 【注2】SGI 用 malloc 而非 new 来分配内存应该有两个原因：
    // 		 其一是历史因素；其二是 C++ 没有相应于 realloc 的内存分配操作。
    static void (*__set_malloc_handler(void (*__f)()))()
    {
        void (*__old)() = __malloc_alloc_oom_handler;
        __malloc_alloc_oom_handler = __f;
        return (__old);
    }
};

// malloc_alloc oom 处理

#ifndef __STL_STATIC_TEMPLATE_MEMBER_BUG
template <int __inst>
void (*__malloc_alloc_template<__inst>::__malloc_alloc_oom_handler)() = 0; // 初值为 0，由客户自行设定
#endif

template <int __inst>
void *__malloc_alloc_template<__inst>::_S_oom_malloc(size_t __n)
{
    void (*__my_malloc_handler)();
    void *__result;

    // 申请大小为 __n 的内存
    for (;;) // 不断尝试释放、分配、再释放、再分配内存
    {
        __my_malloc_handler = __malloc_alloc_oom_handler;       // out-of-memory 处理函数
        if (0 == __my_malloc_handler) { __THROW_BAD_ALLOC; }    // 检查 out-of-memory 处理函数的有效性
        (*__my_malloc_handler)();                               // 调用 out-of-memory 处理函数，企图释放内存
        __result = malloc(__n);                                 // 再次尝试分配内存
        if (__result) return (__result);                        // 如果分配成功，就返回结果，否则重新循环
    }
}

template <int __inst>
void *__malloc_alloc_template<__inst>::_S_oom_realloc(void *__p, size_t __n)
{
    void (*__my_malloc_handler)();
    void *__result;

    // 给一个已经分配了内存的空间地址重新分配空间，参数 __p 为原有的空间地址，__n 是重新申请的内存大小
    for (;;) // 不断尝试释放、分配、再释放、再分配内存
    {
        __my_malloc_handler = __malloc_alloc_oom_handler;       // out-of-memory 处理函数
        if (0 == __my_malloc_handler) {  __THROW_BAD_ALLOC; }   // 检查 out-of-memory 处理函数的有效性
        (*__my_malloc_handler)();                               // 调用 out-of-memory 处理函数，企图释放内存
        __result = realloc(__p, __n);                           // 再次尝试再分配内存
        if (__result) return (__result);                        // 如果分配成功，就返回结果，否则重新循环
    }
}

// 定义 malloc_alloc 为第一级 0 号分配器
typedef __malloc_alloc_template<0> malloc_alloc; // 直接将 “非类型参数 __inst” 指定为 0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
// ----------------------------------------
// SGI STL 两级分配器的统一封装和变形

// 分配器标准接口封装。
// 单纯地转调用，调用传递给分配器(第一级或第二级)；多一层包装，使 _Alloc 具备标准接口。
template <class _Tp, class _Alloc>
class simple_alloc
{

public:
    // 配置 n 个元素
    static _Tp *allocate(size_t __n)
    {
        return 0 == __n ? 0 : (_Tp *)_Alloc::allocate(__n * sizeof(_Tp));
    }
    static _Tp *allocate(void)
    {
        return (_Tp *)_Alloc::allocate(sizeof(_Tp));
    }
    static void deallocate(_Tp *__p, size_t __n)
    {
        if (0 != __n)
            _Alloc::deallocate(__p, __n * sizeof(_Tp));
    }
    static void deallocate(_Tp *__p)
    {
        _Alloc::deallocate(__p, sizeof(_Tp));
    }
};

// 能够记录大小的分配器封装。
// Allocator adaptor to check size arguments for debugging.
// Reports errors using assert.  Checking can be disabled with
// NDEBUG, but it's far better to just use the underlying allocator
// instead when no checking is desired.
// There is some evidence that this can confuse Purify.
template <class _Alloc>
class debug_alloc
{

private:
    enum
    {
        _S_extra = 8
    };  // Size of space used to store size.  Note
        // that this must be large enough to preserve
        // alignment.

public:
    static void *allocate(size_t __n)
    {
        char *__result = (char *)_Alloc::allocate(__n + (int)_S_extra);
        *(size_t *)__result = __n;
        return __result + (int)_S_extra;
    }

    static void deallocate(void *__p, size_t __n)
    {
        char *__real_p = (char *)__p - (int)_S_extra;
        assert(*(size_t *)__real_p == __n);
        _Alloc::deallocate(__real_p, __n + (int)_S_extra);
    }

    static void *reallocate(void *__p, size_t __old_sz, size_t __new_sz)
    {
        char *__real_p = (char *)__p - (int)_S_extra;
        assert(*(size_t *)__real_p == __old_sz);
        char *__result = (char *)
            _Alloc::reallocate(__real_p, __old_sz + (int)_S_extra,
                               __new_sz + (int)_S_extra);
        *(size_t *)__result = __new_sz;
        return __result + (int)_S_extra;
    }
};

// ----------------------------------------
// SGI STL 第二级分配器

#ifdef __USE_MALLOC

// 不使用第二级分配器，为了兼容，补充定义 alloc 和 single_client_alloc，均指定为第一级 0 号分配器
typedef malloc_alloc alloc;
typedef malloc_alloc single_client_alloc;

#else

// Default node allocator.
// With a reasonable compiler, this should be roughly as fast as the
// original STL class-specific allocators, but with less fragmentation.
// Default_alloc_template parameters are experimental and MAY
// DISAPPEAR in the future.  Clients should just use alloc for now.
//
// Important implementation properties:
// 1. If the client request an object of size > _MAX_BYTES, the resulting
//    object will be obtained directly from malloc.
// 2. In all other cases, we allocate an object of size exactly
//    _S_round_up(requested_size).  Thus the client has enough size
//    information that we can return the object to the proper free list
//    without permanently losing part of the object.
//

// The first template parameter specifies whether more than one thread
// may use this allocator.  It is safe to allocate an object from
// one instance of a default_alloc and deallocate it with another
// one.  This effectively transfers its ownership to the second one.
// This may have undesirable effects on reference locality.
// The second parameter is unreferenced and serves only to allow the
// creation of multiple default_alloc instances.
// Node that containers built on different allocator instances have
// different types, limiting the utility of this approach.

// 内存池静态常量
#if defined(__SUNPRO_CC) || defined(__GNUC__)
// breaks if we make these template class members:
enum
{
    _ALIGN = 8
};
enum
{
    _MAX_BYTES = 128
};
enum
{
    _NFREELISTS = 16
}; // _MAX_BYTES/_ALIGN
#endif

// SGI STL 第二级分配器，GCC 默认使用第二级分配器，用于小于 128byte 的内存管理。
// 无 “template 类型参数”，前者用于支持多线程环境，后者用于支持多个实例的创建（但也使容器的 allocator 类型不一致）
template <bool threads, int inst> 
class __default_alloc_template
{

private:
    // 内存池静态常量
    //     Really we should use static const int x = N
    //     instead of enum { x = N }, but few compilers accept the former.【注】
#if !(defined(__SUNPRO_CC) || defined(__GNUC__))
    enum
    {
        _ALIGN = 8
    }; // 小额区块的上调边界
    enum
    {
        _MAX_BYTES = 128
    }; // 小额区块的上限
    enum
    {
        _NFREELISTS = 16
    }; // free-list 的个数 = _MAX_BYTES / _ALIGN
#endif
    // 大小调整函数，将所有小型区块的内存需求量增加至 8 的倍数【技】
    static size_t _S_round_up(size_t __bytes)
    {
        // ~(size_t)_ALIGN - 1) => 11111000，其中 8 及其倍数位是 true，而 8 的余数位是 false，做 & 运算完成截断
        return (((__bytes) + (size_t)_ALIGN - 1) & ~((size_t)_ALIGN - 1));
    }

__PRIVATE :
    // free-list 的节点结构，使用 union 降低维护链表 free-list 带来的额外负担
    union _Obj {
        union _Obj *_M_free_list_link; // 指向相同形式的另一个 _Obj  
        char _M_client_data[1];        // 指向实际区块
    };

private:
    // 16 个自由链表，_S_free_list 是一个二级指针数组，每个元素指向一级指针，
    // 一级指针是指向 _Obj 类型的指针，各自管理大小分别为 8， 16， 24， 32，...128 bytes(8 的倍数)的小额区块。
    // 【注1】__STL_VOLATILE 的含义
    // 		元素 *my_free_list 或 _S_free_list[]可能被优化到寄存器中，从而使库代码无法 lock 住对它的读调用。
    // 		若在寄存器中另一个线程可能会修改该寄存器的值， 若在内存中另一个线程没有访问权力有修改保护。
    // 【注2】__STL_VOLATILE 的用法
    //      用 volatile 修饰声明变量必须在内存中，这里修饰的是 *my_free_list 或 _S_free_list[]，
    // 		是 free_list 数组中的一个元素，而不是数组指针，所以 volatile 放在两个 * 中间。
#if defined(__SUNPRO_CC) || defined(__GNUC__) || defined(__HP_aCC)
    static _Obj *__STL_VOLATILE _S_free_list[];
#else
    static _Obj *__STL_VOLATILE _S_free_list[_NFREELISTS]; // Specifying a size results in duplicate def for 4.1
#endif
    // 下标获取函数，根据申请数据块大小找到相应自由链表的下标，n 从 0 起算
    static size_t _S_freelist_index(size_t __bytes)
    {
        // 1~8 => 0, 9~16 => 1, 17~24 => 2, 25~32 => 3, ...
        return (((__bytes) + (size_t)_ALIGN - 1) / (size_t)_ALIGN - 1);
    }

    // 单对象内存分配函数，基于下面的 _S_chunk_alloc 函数，Returns an object of size __n, and optionally adds to size __n free list.
    static void *_S_refill(size_t __n);
    // 多对象内存分配函数，Allocates a chunk for nobjs of size size. nobjs may be reduced if it is inconvenient to allocate the requested number.
    static char *_S_chunk_alloc(size_t __size, int &__nobjs);

    // 内存状态指针，Chunk allocation state.
    static char *_S_start_free; // 内存池起始位置。只在 _S_chunk_alloc() 中变化
    static char *_S_end_free;   // 内存池结束位置。只在 _S_chunk_alloc() 中变化
    static size_t _S_heap_size; // 内存池的大小，内存池数据结构为堆

#ifdef __STL_THREADS
    // STL 线程锁对象
    static _STL_mutex_lock _S_node_allocator_lock;
#endif
    // 自定义线程锁对象
    // 注：
    //     It would be nice to use _STL_auto_lock here.  But we
    //     don't need the NULL check.  And we do need a test whether
    //     threads have actually been started.
    class _Lock;
    friend class _Lock;
    class _Lock
    {
    public:
        _Lock() { __NODE_ALLOCATOR_LOCK; }
        ~_Lock() { __NODE_ALLOCATOR_UNLOCK; }
    };

public:
    // 内存分配函数，申请大小为 n 的数据块，返回该数据块的起始地址
    // 注：__n must be > 0
    static void *allocate(size_t __n)
    {
        void *__ret = 0;

        if (__n > (size_t)_MAX_BYTES)
        // 如果需求区块大于 128 bytes，就转调用第一级分配器
        {
            __ret = malloc_alloc::allocate(__n);
        }
        else
        // 根据申请空间的大小寻找相应的自由链表（16 个自由链表中的一个）
        {
            _Obj *__STL_VOLATILE *__my_free_list = _S_free_list + _S_freelist_index(__n);
#ifndef _NOTHREADS
            // 获取线程锁，在这里通过构造函数调用获取锁，这可确保在退出或堆栈展开期间释放。
            // Acquire the lock here with a constructor call.
            // This ensures that it is released in exit or during stack unwinding.
            _Lock __lock_instance;
#endif
            _Obj *__RESTRICT __result = *__my_free_list;
            if (__result == 0)
            // 自由链表没有可用数据块，就将区块大小先调整至 8 倍数边界，然后调用 _S_refill() 分配空间
                __ret = _S_refill(_S_round_up(__n));
            else
            // 如果自由链表中有空闲数据块，则取出一个，并把自由链表的指针指向下一个数据块
            {
                // 将 __my_free_list 头节点移出并返回
                *__my_free_list = __result->_M_free_list_link;
                __ret = __result;
            }
        }

        return __ret;
    };

    // 内存释放函数，释放 __p 指向的大小为 __n 的空间
    // 注： __p may not be 0
    static void deallocate(void *__p, size_t __n)
    {
        if (__n > (size_t)_MAX_BYTES)
        // 大于 128 bytes，就调用第一级分配器的释放操作
        {
            malloc_alloc::deallocate(__p, __n);
        }
        else
        // 否则将空间回收到相应自由链表（由释放块的大小决定）中
        {
            _Obj *__STL_VOLATILE *__my_free_list = _S_free_list + _S_freelist_index(__n);
            _Obj *__q = (_Obj *)__p;

#ifndef _NOTHREADS
            // 获取线程锁，acquire lock
            _Lock __lock_instance;
#endif /* _NOTHREADS */
            // 将 __p 按头插法插入到 __my_free_list 当中
            __q->_M_free_list_link = *__my_free_list; // 调整自由链表，回收数据块
            *__my_free_list = __q;
            // lock is released here
        }
    }

    // 内存再分配函数
    static void *reallocate(void *__p, size_t __old_sz, size_t __new_sz);
};

// 定义 alloc 为第二级 0 号分配器
typedef __default_alloc_template<__NODE_ALLOCATOR_THREADS, 0> alloc;
// 定义 single_client_alloc 为第二级单线程 0 号分配器
typedef __default_alloc_template<false, 0> single_client_alloc;      

// 第二级分配器的相等运算符
template <bool __threads, int __inst>
inline bool operator==(const __default_alloc_template<__threads, __inst> &, const __default_alloc_template<__threads, __inst> &)
{
    return true;
}

// 第二级分配器的不等运算符
#ifdef __STL_FUNCTION_TMPL_PARTIAL_ORDER
template <bool __threads, int __inst>
inline bool operator!=(const __default_alloc_template<__threads, __inst> &, const __default_alloc_template<__threads, __inst> &)
{
    return false;
}
#endif /* __STL_FUNCTION_TMPL_PARTIAL_ORDER */

// 多对象内存分配函数的类外实现，从内存池中取空间，不够再考虑分配空间
// 注：假设 __size 已经上调至 8 的倍数
/* We allocate memory in large chunks in order to avoid fragmenting     */
/* the malloc heap too much.                                            */
/* We assume that size is properly aligned.                             */
/* We hold the allocation lock.                                         */
template <bool __threads, int __inst>
char* __default_alloc_template<__threads, __inst>::_S_chunk_alloc(size_t __size, int &__nobjs)
{
    char *__result;                                    // 申请空间的起始指针
    size_t __total_bytes = __size * __nobjs;           // 需要申请空间的大小
    size_t __bytes_left = _S_end_free - _S_start_free; // 计算内存池剩余空间

    if (__bytes_left >= __total_bytes)
    // 内存池剩余空间完全满足申请
    {
        __result = _S_start_free;               // 指向起始地址
        _S_start_free += __total_bytes;         // 内存池缩减
        return (__result);                      // 返回结果
    }
    else if (__bytes_left >= __size)
    // 内存池剩余空间不能满足申请，但能提供一个及以上的区块
    {
        __nobjs = (int)(__bytes_left / __size); // 调整申请区块的个数
        __total_bytes = __size * __nobjs;       // 调整需要申请空间的大小
        __result = _S_start_free;               // 指向起始地址
        _S_start_free += __total_bytes;         // 内存池缩减
        return (__result);                      // 返回结果
    }
    else
    // 内存池剩余空间连一个区块的大小都无法提供
    {
        // 新需求量：原需求量的二倍 + 现有内存池大小的十六分之一并扩充至 8 倍数
        size_t __bytes_to_get = 2 * __total_bytes + _S_round_up(_S_heap_size >> 4);
        // 尝试利用内存池的剩余空间，放到合适的自由链表中
        if (__bytes_left > 0)
        {
            // 找到合适的自由链表（__bytes_left 一定是 8 的倍数）
            _Obj *__STL_VOLATILE *__my_free_list = _S_free_list + _S_freelist_index(__bytes_left);
            // 头插法插入内存池
            ((_Obj *)_S_start_free)->_M_free_list_link = *__my_free_list;
            *__my_free_list = (_Obj *)_S_start_free;
        }
        // 从堆中分配空间，用来补充内存池
        _S_start_free = (char *)malloc(__bytes_to_get);
        // 如果堆空间不足，malloc() 失败
        if (0 == _S_start_free)
        {
            size_t __i;
            _Obj *__STL_VOLATILE *__my_free_list;
            _Obj *__p;
            // 尝试寻找自由链表中的可用内存
            // 注：为了避免多进程机器上的灾难性问题，不会尝试分配较小的区块。
            //     只会寻找自由链表中 “没有用过的、足够大” 的区块来使用。
            for (__i = __size; __i <= (size_t)_MAX_BYTES; __i += (size_t)_ALIGN)
            // 从 __size 这个最小可用需求往大的方向寻找
            {
                // 找到合适的自由链表
                __my_free_list = _S_free_list + _S_freelist_index(__i);
                __p = *__my_free_list;
                if (0 != __p)
                // 如果找到内存池对应的可用区块，递归调用 chunk_alloc 并 return
                {
                    // 调整自由链表，释放可用区块
                    *__my_free_list = __p->_M_free_list_link; // 取出这个区块
                    _S_start_free = (char *)__p;              // 设置内存池起始地址
                    _S_end_free = _S_start_free + __i;        // 设置内存池结束地址
                    // 递归调用 chunk_alloc 取空间，并修正 __nobjs
                    return (_S_chunk_alloc(__size, __nobjs));
                    // Any leftover piece will eventually make it to the right free list.
                }
            }
            // 如果出现意外（山穷水尽，到处都没有内存了）
            _S_end_free = 0;                                                // In case of exception.
            // 调研第一级分配器，尝试 out-of-memory 机制能否尽力
            // 注：这要么导致抛出异常，要么改善内存不足的情况
            _S_start_free = (char *)malloc_alloc::allocate(__bytes_to_get);
        }
        // 到此保证拿到 __bytes_to_get 大小的内存到内存池中
        _S_heap_size += __bytes_to_get;
        _S_end_free = _S_start_free + __bytes_to_get;
        // 递归调用 chunk_alloc 取空间，并修正 __nobjs
        return (_S_chunk_alloc(__size, __nobjs));
    }
}

// 单对象内存分配函数的类外实现，默认调用多对象内存分配函数，分配 20 个对象的冗余空间，冗余的放到自由链表中
// 注：假设 __n 已经上调至 8 的倍数
/* Returns an object of size __n, and optionally adds to size __n free list.*/
/* We assume that __n is properly aligned.                                  */
/* We hold the allocation lock.                                             */
template <bool __threads, int __inst>
void* __default_alloc_template<__threads, __inst>::_S_refill(size_t __n)
{
    // 虽然只要求一个对象，但这里会默认分配 20 个冗余
    int __nobjs = 20;
    // 调用 _S_chunk_alloc()，默认取 20 个区块作为 free list 的新节点
    char *__chunk = _S_chunk_alloc(__n, __nobjs);
    _Obj *__STL_VOLATILE *__my_free_list;
    _Obj *__result;
    _Obj *__current_obj;
    _Obj *__next_obj;
    int __i;

    // 如果只获得一个区块，那么这个区块就直接分给调用者，自由链表中不会增加新节点
    if (1 == __nobjs)
        return (__chunk);
    // 否则根据申请区块的大小找到相应的自由链表
    __my_free_list = _S_free_list + _S_freelist_index(__n);
 
    // 在分配的 chunk 空间内建立 free list
    __result = (_Obj *)__chunk;                             // 这一块准备返回给调用者
    *__my_free_list = __next_obj = (_Obj *)(__chunk + __n); // 这里开始放到自由链表
    for (__i = 1;; __i++)                                   // 将 free list 各结点串接，从 1 开始
    {
        // 记录当前插入的结点
        __current_obj = __next_obj;
        // 记录下一个结点（转换为 char* 前进 __n bytes，再转换为 _Obj* 指向 __n bytes 大小的区块【技】）
        __next_obj = (_Obj *)((char *)__next_obj + __n); 
        if (__nobjs - 1 == __i)
        // 如果 __i 或 __current_obj 指向最后一个 _Obj 或 __next_obj 为 “空”
        { __current_obj->_M_free_list_link = 0;  break; }  // 收尾 
        else
        // 如果还没到最后一个结点
        { __current_obj->_M_free_list_link = __next_obj; } // 衔接
    }
    return (__result);
}

// 内存再分配函数的类外实现，将 __p 指向的大小为 __old_sz 的空间重新分配成大小为 __new_sz 的空间。
template <bool threads, int inst>
void* __default_alloc_template<threads, inst>::reallocate(void *__p, size_t __old_sz, size_t __new_sz)
{
    void *__result;
    size_t __copy_sz;

    // 如果需求区块大于 128 bytes，就直接调用 realloc 分配
    if (__old_sz > (size_t)_MAX_BYTES && __new_sz > (size_t)_MAX_BYTES)
    {
        return (realloc(__p, __new_sz));
    }
    // 如果原空间与新空间大小相等，无须再分配直接返回
    if (_S_round_up(__old_sz) == _S_round_up(__new_sz))
        return (__p);
    // 否则转调用第二级分配器的 allocate，重新分配 __new_sz 大小的空间
    __result = allocate(__new_sz);
    // 拷贝空间
    __copy_sz = __new_sz > __old_sz ? __old_sz : __new_sz; // 确定要拷贝的空间大小，最大为 __old_sz
    memcpy(__result, __p, __copy_sz); // 拷贝 __p 指向的内存到 _result 指向的内存，大小为 __copy_sz
    deallocate(__p, __old_sz); // 释放 __p 指向的内存，大小为 __old_sz
    return (__result); // 返回已再分配的空间指针
}

#ifdef __STL_THREADS
// 类内 STL 线程锁对象
template <bool __threads, int __inst>
_STL_mutex_lock __default_alloc_template<__threads, __inst>::_S_node_allocator_lock __STL_MUTEX_INITIALIZER;
#endif

// 静态成员变量的定义与初值设定
template <bool __threads, int __inst>
char *__default_alloc_template<__threads, __inst>::_S_start_free = 0;

template <bool __threads, int __inst>
char *__default_alloc_template<__threads, __inst>::_S_end_free = 0;

template <bool __threads, int __inst>
size_t __default_alloc_template<__threads, __inst>::_S_heap_size = 0;

template <bool __threads, int __inst>
typename __default_alloc_template<__threads, __inst>::_Obj *__STL_VOLATILE
    __default_alloc_template<__threads, __inst>::_S_free_list[
#if defined(__SUNPRO_CC) || defined(__GNUC__) || defined(__HP_aCC)
        _NFREELISTS
#else
        __default_alloc_template<__threads, __inst>::_NFREELISTS
#endif
] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
};
// The 16 zeros are necessary to make version 4.1 of the SunPro
// compiler happy.  Otherwise it appears to allocate too little
// space for the array.

#endif /* ! __USE_MALLOC */

// ==================================================
// SGI STL 分配器对外接口（standard-conforming allocators，上面的是 SGI-style allocators）

// This implements allocators as specified in the C++ standard.
//
// Note that standard-conforming allocators use many language features
// that are not yet widely implemented.  In particular, they rely on
// member templates, partial specialization, partial ordering of function
// templates, the typename keyword, and the use of the template keyword
// to refer to a template member of a dependent type.

#ifdef __STL_USE_STD_ALLOCATORS

// ----------------------------------------
// SGI STL 分配器 alloc 的封装：allocator

// allocator 类模板，封装 alloc
template <class _Tp>
class allocator
{
    // 类型别名
    typedef alloc _Alloc; // The underlying allocator. 底层 allocator 为第二级分配器
public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef _Tp *pointer;
    typedef const _Tp *const_pointer;
    typedef _Tp &reference;
    typedef const _Tp &const_reference;
    typedef _Tp value_type;

    template <class _Tp1>
    struct rebind
    {
        typedef allocator<_Tp1> other; // 一个模板特例化
    };

    // 拷贝控制函数
    allocator() __STL_NOTHROW {}
    allocator(const allocator &) __STL_NOTHROW {}
    template <class _Tp1>
    allocator(const allocator<_Tp1> &) __STL_NOTHROW {}
    ~allocator() __STL_NOTHROW {}

    // 取地址函数
    pointer address(reference __x) const { return &__x; }
    const_pointer address(const_reference __x) const { return &__x; }

    // 内存分配函数
    // __n is permitted to be 0.  The C++ standard says nothing about what
    // the return value is when __n == 0.
    _Tp *allocate(size_type __n, const void * = 0)
    {
        return __n != 0 ? static_cast<_Tp *>(_Alloc::allocate(__n * sizeof(_Tp))) : 0; // 转调用第二级分配器
    }

    // 内存释放函数
    // __p is not permitted to be a null pointer.
    void deallocate(pointer __p, size_type __n)
    {
        _Alloc::deallocate(__p, __n * sizeof(_Tp)); // 转调用第二级分配器
    }

    // 
    size_type max_size() const __STL_NOTHROW
    {
        // 代表机器允许放入容器的最大元素数目，
        // 通过强制类型转换，size_t(-1) 恰能代表其能表示的最大值，即 2^32 (2^64，64 位机器)。【技】
        return size_t(-1) / sizeof(_Tp);
    }

    // 元素重构与销毁函数
    void construct(pointer __p, const _Tp &__val) { new (__p) _Tp(__val); }
    void destroy(pointer __p) { __p->~_Tp(); }
};

// allocator 类模板的 void 特例化
template <>
class allocator<void>
{
public:
    // 类型别名
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef void *pointer;
    typedef const void *const_pointer;
    typedef void value_type;

    template <class _Tp1>
    struct rebind
    {
        typedef allocator<_Tp1> other;
    };
};

// allocator 类模板的相等运算符
template <class _T1, class _T2>
inline bool operator==(const allocator<_T1> &, const allocator<_T2> &)
{
    return true;
}

// allocator 类模板的不等运算符
template <class _T1, class _T2>
inline bool operator!=(const allocator<_T1> &, const allocator<_T2> &)
{
    return false;
}

// ----------------------------------------
// SGI STL 分配器 alloc 的封装的适配：__allocator

// allocator<_Tp> 的适配器版本，内部实现一模一样，只是增加一个模板参数
// 注：
//     新增参数 _Alloc 用于支持多个实例的创建；
//     默认各个 alloc 类型可能不同；
//     默认 alloc 的成员函数可能非静态；
// Allocator adaptor to turn an SGI-style allocator (e.g. alloc, malloc_alloc)
// into a standard-conforming allocator.   Note that this adaptor does
// *not* assume that all objects of the underlying alloc class are
// identical, nor does it assume that all of the underlying alloc's
// member functions are static member functions.  Note, also, that
// __allocator<_Tp, alloc> is essentially the same thing as allocator<_Tp>.

template <class _Tp, class _Alloc>
struct __allocator
{
    _Alloc __underlying_alloc;

    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef _Tp *pointer;
    typedef const _Tp *const_pointer;
    typedef _Tp &reference;
    typedef const _Tp &const_reference;
    typedef _Tp value_type;

    template <class _Tp1>
    struct rebind
    {
        typedef __allocator<_Tp1, _Alloc> other;
    };

    __allocator() __STL_NOTHROW {}
    __allocator(const __allocator &__a) __STL_NOTHROW : __underlying_alloc(__a.__underlying_alloc) {}
    template <class _Tp1>
    __allocator(const __allocator<_Tp1, _Alloc> &__a) __STL_NOTHROW : __underlying_alloc(__a.__underlying_alloc) {}
    ~__allocator() __STL_NOTHROW {}

    pointer address(reference __x) const { return &__x; }
    const_pointer address(const_reference __x) const { return &__x; }

    // __n is permitted to be 0.
    _Tp *allocate(size_type __n, const void * = 0)
    {
        return __n != 0
                   ? static_cast<_Tp *>(__underlying_alloc.allocate(__n * sizeof(_Tp)))
                   : 0;
    }

    // __p is not permitted to be a null pointer.
    void deallocate(pointer __p, size_type __n)
    {
        __underlying_alloc.deallocate(__p, __n * sizeof(_Tp));
    }

    size_type max_size() const __STL_NOTHROW
    {
        return size_t(-1) / sizeof(_Tp);
    }

    void construct(pointer __p, const _Tp &__val) { new (__p) _Tp(__val); }
    void destroy(pointer __p) { __p->~_Tp(); }
};

// allocator<_Tp> 的适配器版本的 void 特例化
template <class _Alloc>
class __allocator<void, _Alloc>
{
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef void *pointer;
    typedef const void *const_pointer;
    typedef void value_type;

    template <class _Tp1>
    struct rebind
    {
        typedef __allocator<_Tp1, _Alloc> other;
    };
};

// __allocator 类模板的相等运算符
template <class _Tp, class _Alloc>
inline bool operator==(const __allocator<_Tp, _Alloc> &__a1, const __allocator<_Tp, _Alloc> &__a2)
{
    return __a1.__underlying_alloc == __a2.__underlying_alloc;
}

// __allocator 类模板的不等运算符
#ifdef __STL_FUNCTION_TMPL_PARTIAL_ORDER
template <class _Tp, class _Alloc>
inline bool operator!=(const __allocator<_Tp, _Alloc> &__a1, const __allocator<_Tp, _Alloc> &__a2)
{
    return __a1.__underlying_alloc != __a2.__underlying_alloc;
}
#endif /* __STL_FUNCTION_TMPL_PARTIAL_ORDER */

// Comparison operators for all of the predifined SGI-style allocators.
// This ensures that __allocator<malloc_alloc> (for example) will
// work correctly.

// 第一级分配器的相等运算符
template <int inst>
inline bool operator==(const __malloc_alloc_template<inst> &, const __malloc_alloc_template<inst> &)
{
    return true;
}

// 第一级分配器的不等运算符
#ifdef __STL_FUNCTION_TMPL_PARTIAL_ORDER
template <int __inst>
inline bool operator!=(const __malloc_alloc_template<__inst> &, const __malloc_alloc_template<__inst> &)
{
    return false;
}
#endif /* __STL_FUNCTION_TMPL_PARTIAL_ORDER */

// 两级统一分配器（能够记大小）的相等运算符
template <class _Alloc>
inline bool operator==(const debug_alloc<_Alloc> &, const debug_alloc<_Alloc> &)
{
    return true;
}

// 两级统一分配器（能够记大小）的不等运算符
#ifdef __STL_FUNCTION_TMPL_PARTIAL_ORDER
template <class _Alloc>
inline bool operator!=(const debug_alloc<_Alloc> &, const debug_alloc<_Alloc> &)
{
    return false;
}
#endif /* __STL_FUNCTION_TMPL_PARTIAL_ORDER */

// ----------------------------------------
// SGI STL 分配器 alloc 的封装的适配：_Alloc_traits

// 支持 SGI-style allocators 和 standard-conforming allocator；
// 支持单例模式；
// Another allocator adaptor: _Alloc_traits.  This serves two
// purposes.  First, make it possible to write containers that can use
// either SGI-style allocators or standard-conforming allocator.
// Second, provide a mechanism so that containers can query whether or
// not the allocator has distinct instances.  If not, the container
// can avoid wasting a word of memory to store an empty object.

// 对模板参数 _Alloc 的假设；
// This adaptor uses partial specialization.  The general case of
// _Alloc_traits<_Tp, _Alloc> assumes that _Alloc is a
// standard-conforming allocator, possibly with non-equal instances
// and non-static members.  (It still behaves correctly even if _Alloc
// has static member and if all instances are equal.  Refinements
// affect performance, not correctness.)

// 两个通用成员；
// There are always two members: allocator_type, which is a standard-
// conforming allocator type for allocating objects of type _Tp, and
// _S_instanceless, a static const member of type bool.  If
// _S_instanceless is true, this means that there is no difference
// between any two instances of type allocator_type.  Furthermore, if
// _S_instanceless is true, then _Alloc_traits has one additional
// member: _Alloc_type.  This type encapsulates allocation and
// deallocation of objects of type _Tp through a static interface; it
// has two member functions, whose signatures are
//    static _Tp* allocate(size_t)
//    static void deallocate(_Tp*, size_t)

// The fully general version.

template <class _Tp, class _Allocator>
struct _Alloc_traits
{
    static const bool _S_instanceless = false;
    typedef typename _Allocator::__STL_TEMPLATE rebind<_Tp>::other allocator_type;
};

template <class _Tp, class _Allocator>
const bool _Alloc_traits<_Tp, _Allocator>::_S_instanceless;

// The version for the default allocator.

template <class _Tp, class _Tp1>
struct _Alloc_traits<_Tp, allocator<_Tp1>>
{
    static const bool _S_instanceless = true;
    typedef simple_alloc<_Tp, alloc> _Alloc_type;
    typedef allocator<_Tp> allocator_type;
};

// Versions for the predefined SGI-style allocators.

template <class _Tp, int __inst>
struct _Alloc_traits<_Tp, __malloc_alloc_template<__inst>>
{
    static const bool _S_instanceless = true;
    typedef simple_alloc<_Tp, __malloc_alloc_template<__inst>> _Alloc_type;
    typedef __allocator<_Tp, __malloc_alloc_template<__inst>> allocator_type;
};

template <class _Tp, bool __threads, int __inst>
struct _Alloc_traits<_Tp, __default_alloc_template<__threads, __inst>>
{
    static const bool _S_instanceless = true;
    typedef simple_alloc<_Tp, __default_alloc_template<__threads, __inst>>
        _Alloc_type;
    typedef __allocator<_Tp, __default_alloc_template<__threads, __inst>>
        allocator_type;
};

template <class _Tp, class _Alloc>
struct _Alloc_traits<_Tp, debug_alloc<_Alloc>>
{
    static const bool _S_instanceless = true;
    typedef simple_alloc<_Tp, debug_alloc<_Alloc>> _Alloc_type;
    typedef __allocator<_Tp, debug_alloc<_Alloc>> allocator_type;
};

// Versions for the __allocator adaptor used with the predefined
// SGI-style allocators.

template <class _Tp, class _Tp1, int __inst>
struct _Alloc_traits<_Tp, __allocator<_Tp1, __malloc_alloc_template<__inst>>>
{
    static const bool _S_instanceless = true;
    typedef simple_alloc<_Tp, __malloc_alloc_template<__inst>> _Alloc_type;
    typedef __allocator<_Tp, __malloc_alloc_template<__inst>> allocator_type;
};

template <class _Tp, class _Tp1, bool __thr, int __inst>
struct _Alloc_traits<_Tp, __allocator<_Tp1, __default_alloc_template<__thr, __inst>>>
{
    static const bool _S_instanceless = true;
    typedef simple_alloc<_Tp, __default_alloc_template<__thr, __inst>> _Alloc_type;
    typedef __allocator<_Tp, __default_alloc_template<__thr, __inst>> allocator_type;
};

template <class _Tp, class _Tp1, class _Alloc>
struct _Alloc_traits<_Tp, __allocator<_Tp1, debug_alloc<_Alloc>>>
{
    static const bool _S_instanceless = true;
    typedef simple_alloc<_Tp, debug_alloc<_Alloc>> _Alloc_type;
    typedef __allocator<_Tp, debug_alloc<_Alloc>> allocator_type;
};

#endif /* __STL_USE_STD_ALLOCATORS */

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1174
#endif

__STL_END_NAMESPACE // 命名空间 宏：}

#undef __PRIVATE

#endif /* __SGI_STL_INTERNAL_ALLOC_H */
