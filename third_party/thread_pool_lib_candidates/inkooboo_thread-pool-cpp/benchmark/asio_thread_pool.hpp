#ifndef ASIO_THREAD_POOL_HPP
#define ASIO_THREAD_POOL_HPP

#include <boost/asio.hpp>

#include <functional>
#include <thread>
#include <vector>
#include <memory>

class AsioThreadPool
{
public:
    inline AsioThreadPool(size_t threads);

    inline ~AsioThreadPool()
    {
        stop();
    }

    inline void joinThreadPool();

    template <typename Handler>
    inline void post(Handler &&handler)
    {
        m_io_svc.post(handler);
    }

private:
    inline void start();
    inline void stop();
    inline void worker_thread_func();

    boost::asio::io_service m_io_svc;
    std::unique_ptr<boost::asio::io_service::work> m_work;

    std::vector<std::thread> m_threads;
};

inline AsioThreadPool::AsioThreadPool(size_t threads)
    : m_threads(threads)
{
    start();
}

inline void AsioThreadPool::start()
{
    m_work.reset(new boost::asio::io_service::work(m_io_svc));

    for (auto &i : m_threads)
    {
        i = std::thread(&AsioThreadPool::worker_thread_func, this);
    }

}

inline void AsioThreadPool::stop()
{
    m_work.reset();

    m_io_svc.stop();

    for (auto &i : m_threads)
    {
        if (i.joinable())
        {
            i.join();
        }
    }
}

inline void AsioThreadPool::joinThreadPool()
{
    m_io_svc.run();
}

inline void AsioThreadPool::worker_thread_func()
{
    joinThreadPool();
}

#endif
