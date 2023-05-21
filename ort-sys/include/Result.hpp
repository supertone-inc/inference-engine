#pragma once

#include "Error.hpp"

enum class ResultCode
{
    Ok = 0,
    Error = -1,
};

template <typename T>
struct Result
{
    ResultCode code;
    T value;
    Error error;
};