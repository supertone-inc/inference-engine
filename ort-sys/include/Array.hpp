#pragma once

#include <cstddef>

template <typename T>
struct Array
{
    const T *data;
    size_t size;
};
