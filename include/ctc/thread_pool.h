/**
 * 固定线程池 — C++17 标准库实现
 *
 * 设计参考 C++ Concurrency in Action（Anthony Williams）第 9 章。
 * 固定数量 worker 线程，共享任务队列，mutex + condition_variable 同步。
 *
 * 不做 work-stealing：批量 CTC 解码中每条语音是独立的长任务（ms 级），
 * 单队列互斥锁持有时间（μs 级）远小于任务执行时间，work-stealing 的
 * 额外复杂度在此场景没有收益。
 *
 * 线程安全：
 *   - Enqueue 可从任意线程调用
 *   - 析构函数等待所有已入队任务完成后才返回（RAII join）
 *   - stop_ 和 active_tasks_ 使用 std::atomic 保证可见性
 */

#ifndef CTC_THREAD_POOL_H_
#define CTC_THREAD_POOL_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace ctc {

class ThreadPool {
public:
  /**
   * 创建线程池
   *
   * @param num_threads worker 线程数量，0 表示自动取硬件并发数
   *                   （std::thread::hardware_concurrency()），最小为 1
   */
  explicit ThreadPool(size_t num_threads = 0);

  ~ThreadPool();

  // 禁止拷贝和移动（std::thread 不可拷贝）
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;
  ThreadPool(ThreadPool &&) = delete;
  ThreadPool &operator=(ThreadPool &&) = delete;

  /**
   * 向任务队列提交一个可调用对象，返回 std::future 用于获取结果
   *
   * 调用者通过 future.get() 阻塞等待任务完成。
   * 若任务抛出异常，异常会被 packaged_task 捕获并在 future.get() 时重新抛出。
   * packaged_task: 把一个函数包装成可以自动保存返回值的任务
   *
   * 示例：
   *   auto future = pool.Enqueue([](int x) { return x * 2; }, 21);
   *   int result = future.get();  // 42
   */
  template <typename F, typename... Args>
  auto Enqueue(F &&f,
               Args &&...args) -> std::future<std::invoke_result_t<F, Args...>>;

  /** 返回 worker 线程数量 */
  size_t NumThreads() const { return workers_.size(); }

  /** 返回当前正在执行的任务数（近似值，用于负载观测） */
  int ActiveTasks() const {
    return active_tasks_.load(std::memory_order_relaxed);
  }

private:
  /** 每个 worker 线程的主循环 */
  void WorkerLoop();

  std::vector<std::thread> workers_;
  std::queue<std::packaged_task<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};
  std::atomic<int> active_tasks_{0};
};

/* ═══════════════════════════════════════════════════════════════
 *  Enqueue 模板实现（必须在头文件中，因为模板实例化需要完整定义）
 * ═══════════════════════════════════════════════════════════════ */

template <typename F, typename... Args>
auto ThreadPool::Enqueue(F &&f, Args &&...args)
    -> std::future<std::invoke_result_t<F, Args...>> {
  using ReturnType = std::invoke_result_t<F, Args...>;

  // 用 shared_ptr 包装 packaged_task，因为 packaged_task 不可拷贝，
  // 而 std::function 要求可拷贝。shared_ptr 也避免了堆上的额外拷贝。
  auto task = std::make_shared<std::packaged_task<ReturnType()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<ReturnType> result =
      task->get_future(); // 把 task 的结果绑定到 future 上

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (stop_)
      throw std::runtime_error("ThreadPool::Enqueue: pool has been stopped");

    // 将 packaged_task 包装为 void() 任务放入队列
    // lambda 捕获 shared_ptr，确保 task 在下沉到 void 后仍然存活
    tasks_.emplace([task]() { (*task)(); });
  }

  cv_.notify_one(); // 唤醒一个等待的 worker
  return result;
}

} // namespace ctc

#endif // CTC_THREAD_POOL_H_
