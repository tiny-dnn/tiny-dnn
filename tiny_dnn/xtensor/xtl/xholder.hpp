/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_HOLDER_HPP
#define XTL_HOLDER_HPP

#include "xcrtp.hpp"
#include "xtl_config.hpp"

namespace xtl
{

    template <class X>
    struct xholder_inner_types;

    template <class D>
    class xholder_base
    {
    public:

        using derived_type = D;

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const & noexcept;
        derived_type derived_cast() && noexcept;

    protected:

        xholder_base() = default;
        ~xholder_base() = default;

        xholder_base(const xholder_base&) = default;
        xholder_base& operator=(const xholder_base&) = default;

        xholder_base(xholder_base&&) = default;
        xholder_base& operator=(xholder_base&&) = default;
    };

    template <template <class> class CRTP>
    class xholder;

    namespace detail
    {
        template <template <class> class CRTP>
        class xholder_impl;
    }

    template <class CRTP>
    struct xholder_inner_types<xholder<CRTP>>
    {
        using implementation_type = detail::xholder_impl<CRTP>;

    };

    template <template <class> class CRTP>
    class xholder
    {
    public:

        using self_type = xholder<CRTP>;
        using implementation_type = typename xholder_inner_type<self_type>::implementation_type;


        xholder();
        ~xholder();
        xholder(const xholder& rhs);
        xholder(xholder&& rhs);
        template <class D>
        xholder(const CRTP<D>& rhs);
        template <class D>
        xholder(CRTP<D>&& rhs);
        xholder(implementation_type* holder);

        xholder& operator=(const xholder& rhs);
        xholder& operator=(xholder&& rhs);

        template <class D>
        xholder& operator=(const CRTP<D>& rhs);
        template <class D>
        xholder& operator=(CRTP<D>&& rhs);

        void swap(xholder& rhs);

        //void display() const;
        //xeus::xguid id() const;

        template <class D>
        D& get() &;
        template <class D>
        const D& get() const &;

    private:

        implementation_type* p_holder;
    };

    template <template <class> class CRTP>
    inline void swap(xholder<CRTP>& lhs, xholder<CRTP>& rhs)
    {
        lhs.swap(rhs);
    }

    namespace detail
    {
        template <template <class> class CRTP>
        class xholder_impl
        {
        public:
      
            xholder_impl() = default;
            xholder_impl(xholder_impl&&) = delete;
            xholder_impl& operator=(const xholder_impl&) = delete;
            xholder_impl& operator=(xholder_impl&&) = delete;
            virtual xholder_impl* clone() const = 0;
            virtual ~xholder_impl() = default;
            virtual bool owning() const = 0;

        protected:

            xholder_impl(const xholder_impl&) = default;
        };

        template <template <class> class CRTP, class D>
        class xholder_owning : public xholder_impl<CRTP>
        {
        public:
        
            using base_type = xholder_impl<CRTP>;

            xholder_owning(const CRTP<D>& value)
                : base_type(),
                  m_value(derived_cast(value))
            {
            }

            xholder_owning(CRTP<D>&& value)
                : base_type(),
                  m_value(derived_cast(std::move(value)))
            {
            }

            virtual ~xholder_owning()
            {
            }

            virtual base_type* clone() const override
            {
                return new xholder_owning(*this);
            }

            inline D& value() & noexcept { return m_value; }
            inline const D& value() const & noexcept { return m_value; }
            inline D value() && noexcept { return m_value; }

            virtual bool owning() const override
            {
                return true;
            }

        private:

            xholder_owning(const xholder_owning&) = default;
            D m_value;
        };
    }

    template <template <class> class CRTP>
    xholder<CRTP>::xholder()
        : p_holder(nullptr)
    {
    }

    template <template <class> class CRTP>
    xholder<CRTP>::xholder(detail::xholder_impl<CRTP>* holder)
        : p_holder(holder)
    {
    }

    template <template <class> class CRTP>
    xholder<CRTP>::~xholder()
    {
        delete p_holder;
    }

    template <template <class> class CRTP>
    xholder<CRTP>::xholder(const xholder& rhs)
        : p_holder(rhs.p_holder ? rhs.p_holder->clone() : nullptr)
    {
    }

    template <template <class> class CRTP>
    template <class D>
    xholder<CRTP>::xholder(CRTP<D>&& rhs)
        : xholder(make_owning_holder(std::move(rhs)))
    {
    }

    template <template <class> class CRTP>
    xholder<CRTP>::xholder(xholder&& rhs)
        : p_holder(rhs.p_holder)
    {
        rhs.p_holder = nullptr;
    }

    template <template <class> class CRTP>
    xholder<CRTP>& xholder<CRTP>::operator=(const xholder& rhs)
    {
        using std::swap;
        xholder tmp(rhs);
        swap(*this, tmp);
        return *this;
    }

    template <template <class> class CRTP>
    xholder<CRTP>& xholder<CRTP>::operator=(xholder&& rhs)
    {
        using std::swap;
        xholder tmp(std::move(rhs));
        swap(*this, tmp);
        return *this;
    }

    template <template <class> class CRTP>
    template <class D>
    xholder<CRTP>& xholder<CRTP>::operator=(CRTP<D>&& rhs)
    {
        using std::swap;
        xholder<CRTP> tmp(make_owning_holder(std::move(rhs)));
        swap(tmp, *this);
        return *this;
    }

    template <template <class> class CRTP>
    void xholder<CRTP>::swap(xholder& rhs)
    {
        std::swap(p_holder, rhs.p_holder);
    }

    template <template <class> class CRTP, class D>
    xholder<CRTP> make_owning_holder(const CRTP<D>& value)
    {
        return xholder<CRTP>(new detail::xholder_owning<CRTP, D>(value));
    }
}

#endif
