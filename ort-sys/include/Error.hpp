#pragma once

class Error
{
public:
    Error();
    Error(const char *message);

    Error(Error &&);
    Error &operator=(Error &&);

    Error(const Error &) = delete;
    Error &operator=(const Error &) = delete;

    virtual ~Error();

    const char *get_message() const;

private:
    class Impl;
    Impl *impl = nullptr;
};
