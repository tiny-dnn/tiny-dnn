/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <string>

namespace xt
{
    // see http://stackoverflow.com/a/20170989
    struct static_string
    {
        template <std::size_t N>
        constexpr static_string(const char (&a)[N]) noexcept
            : data(a), size(N - 1)
        {
        }

        constexpr static_string(const char* a, const std::size_t sz) noexcept
            : data(a), size(sz)
        {
        }

        const char* const data;
        const std::size_t size;
    };

    template <class T>
    constexpr static_string type_name()
    {
#ifdef __clang__
        static_string p = __PRETTY_FUNCTION__;
        return static_string(p.data + 31, p.size - 31 - 1);
#elif defined(__GNUC__)
        static_string p = __PRETTY_FUNCTION__;
#if __cplusplus < 201402
        return static_string(p.data + 36, p.size - 36 - 1);
#else
        return static_string(p.data + 46, p.size - 46 - 1);
#endif
#elif defined(_MSC_VER)
        static_string p = __FUNCSIG__;
        return static_string(p.data + 38, p.size - 38 - 7);
#endif
    }

    template <class T>
    constexpr std::string type_to_string()
    {
        static_string static_name = type_name<T>();
        return std::string(static_name.data, static_name.size);
    }

    template <class T>
    std::string info(const T& t)
    {
        std::string s;
        using shape_type = typename T::shape_type;
        if (detail::is_array<shape_type>::value)
        {
            s += "Type: xtensor, fixed dimension " + std::to_string(t.dimension());
        }
        else
        {
            s += "Type: xarray, dimension " + std::to_string(t.dimension());
        }
        s += "\nValue type: " + type_to_string<typename T::value_type>();
        s += "\nLayout: ";
        if (t.layout() == layout_type::row_major)
        {
            s += "row_major";
        }
        else if (t.layout() == layout_type::column_major)
        {
            s += "column_major";
        }
        else if (t.layout() == layout_type::dynamic)
        {
            s += "dynamic";
        }
        else
        {
            s += "any";
        }
        s += "\nShape: (";
        bool first = true;
        for (const auto& el : t.shape())
        {
            if (!first)
            {
                s += ", ";
            }
            first = false;
            s += std::to_string(el);
        }
        s += ")\nStrides: (";
        first = true;
        for (const auto& el : t.strides())
        {
            if (!first)
            {
                s += ", ";
            }
            first = false;
            s += std::to_string(el);
        }
        s += ")\nSize: " + std::to_string(t.size()) + "\n";
        return s;
    }
}