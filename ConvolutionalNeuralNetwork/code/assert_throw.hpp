#pragma once
#include <stdexcept>

#ifdef NDEBUG

#define assert_throw(expression, msg) ((void)0)
#define assert_throw(expression) ((void)0)

#else

#define assert_throw(expression, msg) if(!expression) { throw std::runtime_error(msg); }										
#define assert_throw(expression) if(!expression) { throw std::runtime_error(""); }

#endif
