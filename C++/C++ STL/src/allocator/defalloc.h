/*
 *
 * Copyright (c) 1994
 * Hewlett-Packard Company
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Hewlett-Packard Company makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 *
 */

// Inclusion of this file is DEPRECATED.  This is the original HP
// default allocator.  It is provided only for backward compatibility.
// This file WILL BE REMOVED in a future release.
//
// DO NOT USE THIS FILE unless you have an old container implementation
// that requires an allocator with the HP-style interface.
//
// Standard-conforming allocators have a very different interface.  The
// standard default allocator is declared in the header <memory>.
//
// 这是原始的 HP default allocator，提供它只是为了回溯兼容。

#ifndef DEFALLOC_H
#define DEFALLOC_H

#include <new.h>
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream.h>
#include <algobase.h>

// 空间分配函数
template <class T>
inline T *allocate(ptrdiff_t size, T *)
{
    // 为了卸载目前的内存分配异常处理函数，强制 C++ 在内存不够的时候抛出 std:bad_alloc
    set_new_handler(0);
    // 申请 size 个 T 类型大小的空间
    T *tmp = (T *)(::operator new((size_t)(size * sizeof(T))));
    if (tmp == 0)
    {
        cerr << "out of memory" << endl;
        exit(1);
    }
    return tmp;
}

// 空间销毁函数
template <class T>
inline void deallocate(T *buffer)
{
    ::operator delete(buffer);
}

// allocator 类模板
template <class T>
class allocator
{
public:
    // 类型成员
    typedef T value_type;
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef T &reference;
    typedef const T &const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    // 内存分配操作，分配 n 个 T 对象的空间
    pointer allocate(size_type n)
    {
        return ::allocate((difference_type)n, (pointer)0);
    }
    // 内存销毁操作，释放 n 个 T 对象的空间
    void deallocate(pointer p)
    {
        ::deallocate(p);
    }
    // 地址获取操作，返回某个对象的地址
    pointer address(reference x)
    {
        return (pointer)&x;
    }
    // 地址获取操作，返回某个 const 对象的地址
    const_pointer const_address(const_reference x)
    {
        return (const_pointer)&x;
    }
    // 初始元素数目
    size_type init_page_size()
    {
        return max(size_type(1), size_type(4096 / sizeof(T)));
    }
    // 返回可容纳的最大元素数目
    size_type max_size() const
    {
        return max(size_type(1), size_type(UINT_MAX / sizeof(T)));
    }
};

// allocator 的类模板实例化 【用意？】
class allocator<void>
{
public:
    typedef void *pointer;
};

#endif
