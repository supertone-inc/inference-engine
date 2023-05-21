#pragma once

#include <string>

struct Error
{
    std::string message;

    const char *get_message() const;
};
