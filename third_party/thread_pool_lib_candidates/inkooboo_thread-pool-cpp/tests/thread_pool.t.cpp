#include <thread_pool/thread_pool.hpp>
#include <test.hpp>

#include <future>
#include <functional>
#include <sstream>
#include <thread>
#include <tuple>

using namespace tp;

using ThreadPoolStd = ThreadPool<>;

int main()
{
    std::cout << "*** Testing TP ***" << std::endl;

    doTest("post job", []()
        {
            ThreadPoolStd pool;

            std::packaged_task<int()> t([]()
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    return 42;
                });

            std::future<int> r = t.get_future();

            pool.post(t);

            ASSERT(42 == r.get());
        });
}
