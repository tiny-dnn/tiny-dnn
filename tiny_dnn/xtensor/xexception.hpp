/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XEXCEPTION_HPP
#define XEXCEPTION_HPP

#include <exception>
#include <iterator>
#include <sstream>
#include <string>

namespace xt
{

    /*******************
     * broadcast_error *
     *******************/

    class broadcast_error : public std::exception
    {
    public:

        template <class S1, class S2>
        broadcast_error(const S1& lhs, const S2& rhs);

        virtual const char* what() const noexcept;

    private:

        std::string m_message;
    };

    /**********************************
     * broadcast_error implementation *
     **********************************/

    template <class S1, class S2>
    inline broadcast_error::broadcast_error(const S1& lhs, const S2& rhs)
    {
        std::ostringstream buf("Incompatible dimension of arrays:", std::ios_base::ate);

        buf << "\n LHS shape = (";
        using size_type1 = typename S1::value_type;
        std::ostream_iterator<size_type1> iter1(buf, ", ");
        std::copy(lhs.cbegin(), lhs.cend(), iter1);

        buf << ")\n RHS shape = (";
        using size_type2 = typename S2::value_type;
        std::ostream_iterator<size_type2> iter2(buf, ", ");
        std::copy(rhs.cbegin(), rhs.cend(), iter2);
        buf << ")";

        m_message = buf.str();
    }

    inline const char* broadcast_error::what() const noexcept
    {
        return m_message.c_str();
    }

    /*******************
     * transpose_error *
     *******************/

    class transpose_error : public std::exception
    {
    public:

        transpose_error(const std::string& msg);

        virtual const char* what() const noexcept;

    private:

        std::string m_message;
    };

    /**********************************
     * transpose_error implementation *
     **********************************/

    inline transpose_error::transpose_error(const std::string& msg)
        : m_message(msg) {}

    inline const char* transpose_error::what() const noexcept
    {
        return m_message.c_str();
    }

    /***************
     * check_index *
     ***************/

    template <class S, class... Args>
    void check_index(const S& shape, Args... args);

    template <class S, class It>
    void check_element_index(const S& shape, It first, It last);

    namespace detail
    {
        template <class S, std::size_t dim>
        inline void check_index_impl(const S&)
        {
        }

        template <class S, std::size_t dim, class... Args>
        inline void check_index_impl(const S& shape, std::size_t arg, Args... args)
        {
            if (sizeof...(Args) + 1 > shape.size())
            {
                check_index_impl<S, dim>(shape, args...);
            }
            else
            {
                if (arg >= shape[dim] && shape[dim] != 1)
                {
                    throw std::out_of_range("index " + std::to_string(arg) + " is out of bounds for axis "
                        + std::to_string(dim) + " with size " + std::to_string(shape[dim]));
                }
                check_index_impl<S, dim + 1>(shape, args...);
            }
        }
    }

    template <class S, class... Args>
    inline void check_index(const S& shape, Args... args)
    {
        detail::check_index_impl<S, 0>(shape, args...);
    }

    template <class S, class It>
    inline void check_element_index(const S& shape, It first, It last)
    {
        auto dst = static_cast<typename S::size_type>(last - first);
        It efirst = last - std::min(shape.size(), dst);
        std::size_t axis = 0;
        while (efirst != last)
        {
            if (*efirst >= shape[axis] && shape[axis] != 1)
            {
                throw std::out_of_range("index " + std::to_string(*efirst) + " is out of bounds for axis "
                    + std::to_string(axis) + " with size " + std::to_string(shape[axis]));
            }
            ++efirst, ++axis;
        }
    }

#ifdef XTENSOR_ENABLE_ASSERT
#define XTENSOR_ASSERT(expr) XTENSOR_ASSERT_IMPL(expr, __FILE__, __LINE__)
#define XTENSOR_ASSERT_IMPL(expr, file, line)                                                                                    \
    try                                                                                                                          \
    {                                                                                                                            \
        expr;                                                                                                                    \
    }                                                                                                                            \
    catch (std::exception & e)                                                                                                   \
    {                                                                                                                            \
        throw std::runtime_error(std::string(file) + ':' + std::to_string(line)                                                  \
            + ": check failed\n\t" + std::string(e.what()));                                                                     \
    }
#else
#define XTENSOR_ASSERT(expr)
#endif
}
#endif
