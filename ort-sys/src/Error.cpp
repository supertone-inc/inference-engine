#include "Error.hpp"

#include <string>

struct Error::Impl
{
    std::string message;
};

Error::Error()
    : impl(nullptr)
{
}

Error::Error(const char *message)
    : impl(new Impl{message})
{
}

Error::Error(Error &&other)
{
    *this = std::move(other);
}

Error &Error::operator=(Error &&other)
{
    if (&other != this)
    {
        if (impl)
        {
            delete impl;
        }

        impl = other.impl;
        other.impl = nullptr;
    }

    return *this;
}

Error::~Error()
{
    if (impl)
    {
        delete impl;
    }
}

const char *Error::get_message() const
{
    return impl->message.c_str();
}
