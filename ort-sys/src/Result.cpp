#include "Result.hpp"

template <typename T>
Result<T> ok(T &&value)
{
    return {ResultCode::Ok, std::move(value), {}};
}

template <typename T>
Result<T> err(Error &&error)
{
    return {ResultCode::Error, {}, std::move(error)};
}
