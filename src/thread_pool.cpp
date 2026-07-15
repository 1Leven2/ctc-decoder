/**
 * 固定线程池 — C++17 标准库实现
 *
 * 每个 worker 线程运行 WorkerLoop：
 *   1. 等待任务队列非空或 stop 信号
 *   2. 取出队首任务
 *   3. 在锁外执行任务（避免阻塞其他 worker）
 *   4. 递减 active_tasks_ 计数器
 *
 * 析构保证：所有已入队任务执行完毕后线程才退出。
 */

#include "ctc/thread_pool.h"

namespace ctc {

ThreadPool::ThreadPool(size_t num_threads) {
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
      num_threads = 1; // hardware_concurrency 不可用时兜底
  }

  workers_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    workers_.emplace_back(&ThreadPool::WorkerLoop, this);
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  cv_.notify_all(); // 唤醒所有等待的 worker

  // join 所有线程——worker 只有在 stop_==true 且队列为空时才退出，
  // 因此析构完成后所有已入队任务保证执行完毕
  for (std::thread &worker : workers_) {
    if (worker.joinable())
      worker.join();
  }
}

void ThreadPool::WorkerLoop() {
  while (true) {
    std::packaged_task<void()> task;

    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      // 等待队列非空或收到 stop 信号
      cv_.wait(lock, [this] {
        return stop_ || !tasks_.empty();
      }); // stop_ == false && tasks_.empty() →释放锁，阻塞等待

      // 收到 stop 且队列为空 → 退出
      if (stop_ && tasks_.empty())
        return;

      // 取出队首任务
      task = std::move(tasks_.front());
      tasks_.pop();
      active_tasks_.fetch_add(1, std::memory_order_relaxed);
    } // 释放锁后再执行任务——避免长时间占用互斥锁

    task(); // 在锁外执行

    active_tasks_.fetch_sub(1, std::memory_order_relaxed);
  }
}

} // namespace ctc
