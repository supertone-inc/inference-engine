#include "Error.hpp"

const char *Error::get_message() const
{
    return message.c_str();
}
