#include "Result.hpp"

template <typename T>
Result<T> ok(T &&value)
{
    return {ResultCode::Ok, value, {}};
}

template <typename T>
Result<T> err(Error &&error)
{
    return {ResultCode::Error, {}, std::move(error)};
}
