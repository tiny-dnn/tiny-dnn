//#define WITHOUT_ASIO 1

#include <thread_pool.hpp>

#ifndef WITHOUT_ASIO
#include <asio_thread_pool.hpp>
#endif

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <future>

using namespace tp;

using ThreadPoolStd = ThreadPool<>;

static const size_t CONCURRENCY = 16;
static const size_t REPOST_COUNT = 1000000;

struct Heavy
{
    bool verbose;
    std::vector<char> resource;

    Heavy(bool verbose = false) : verbose(verbose), resource(100 * 1024 * 1024)
    {
        if(verbose)
        {
            std::cout << "heavy default constructor" << std::endl;
        }
    }

    Heavy(const Heavy& o) : verbose(o.verbose), resource(o.resource)
    {
        if(verbose)
        {
            std::cout << "heavy copy constructor" << std::endl;
        }
    }

    Heavy(Heavy&& o) : verbose(o.verbose), resource(std::move(o.resource))
    {
        if(verbose)
        {
            std::cout << "heavy move constructor" << std::endl;
        }
    }

    Heavy& operator==(const Heavy& o)
    {
        verbose = o.verbose;
        resource = o.resource;
        if(verbose)
        {
            std::cout << "heavy copy operator" << std::endl;
        }
        return *this;
    }

    Heavy& operator==(const Heavy&& o)
    {
        verbose = o.verbose;
        resource = std::move(o.resource);
        if(verbose)
        {
            std::cout << "heavy move operator" << std::endl;
        }
        return *this;
    }

    ~Heavy()
    {
        if(verbose)
        {
            std::cout << "heavy destructor. "
                      << (resource.size() ? "Owns resource"
                                          : "Doesn't own resource")
                      << std::endl;
        }
    }
};


struct RepostJob
{
    // Heavy heavy;

    ThreadPoolStd* thread_pool;
#ifndef WITHOUT_ASIO
    AsioThreadPool* asio_thread_pool;
#endif

    volatile size_t counter;
    long long int begin_count;
    std::promise<void>* waiter;

    RepostJob(ThreadPoolStd* thread_pool, std::promise<void>* waiter)
        : thread_pool(thread_pool)
#ifndef WITHOUT_ASIO
          ,
          asio_thread_pool(0)
#endif
          ,
          counter(0), waiter(waiter)
    {
        begin_count = std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count();
    }

#ifndef WITHOUT_ASIO
    RepostJob(AsioThreadPool* asio_thread_pool, std::promise<void>* waiter)
        : thread_pool(0), asio_thread_pool(asio_thread_pool), counter(0),
          waiter(waiter)
    {
        begin_count = std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count();
    }
#endif

    void operator()()
    {
        if(counter++ < REPOST_COUNT)
        {
#ifndef WITHOUT_ASIO
            if(asio_thread_pool)
            {
                asio_thread_pool->post(*this);
                return;
            }
#endif
            if(thread_pool)
            {
                thread_pool->post(*this);
                return;
            }
        }
        else
        {
            long long int end_count = std::chrono::high_resolution_clock::now()
                                          .time_since_epoch()
                                          .count();
            std::cout << "reposted " << counter << " in "
                      << (double)(end_count - begin_count) / (double)1000000
                      << " ms" << std::endl;
            waiter->set_value();
        }
    }
};

int main(int, const char* [])
{
    std::cout << "Benchmark job reposting" << std::endl;

    {
        std::cout << "***thread pool cpp***" << std::endl;

        std::promise<void> waiters[CONCURRENCY];
        ThreadPoolStd thread_pool;
        for(auto& waiter : waiters)
        {
            thread_pool.post(RepostJob(&thread_pool, &waiter));
        }

        for(auto& waiter : waiters)
        {
            waiter.get_future().wait();
        }
    }

#ifndef WITHOUT_ASIO
    {
        std::cout << "***asio thread pool***" << std::endl;

        size_t workers_count = std::thread::hardware_concurrency();
        if(0 == workers_count)
        {
            workers_count = 1;
        }

        AsioThreadPool asio_thread_pool(workers_count);

        std::promise<void> waiters[CONCURRENCY];
        for(auto& waiter : waiters)
        {
            asio_thread_pool.post(RepostJob(&asio_thread_pool, &waiter));
        }

        for(auto& waiter : waiters)
        {
            waiter.get_future().wait();
        }
    }
#endif

    return 0;
}
