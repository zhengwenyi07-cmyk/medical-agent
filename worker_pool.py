"""消息队列 + 多进程 Worker 池 — 高并发生产级架构

架构：
  FastAPI Server (server.py)
      │
      ▼  push task to Redis / multiprocessing Queue
  ┌─────────────────────────────────────────┐
  │         Task Queue (Redis / IPC)         │
  └─────────────────────────────────────────┘
      │         │         │
      ▼         ▼         ▼
  Worker 0  Worker 1  Worker 2   ...  Worker N-1
      │         │         │
      └─────────┴─────────┘
                │
                ▼  write results
  ┌─────────────────────────────────────────┐
  │         Result Store (Redis / Dict)      │
  └─────────────────────────────────────────┘

Worker 数量 = CPU 核数 × 2（兼顾 CPU 密集和 I/O 密集任务）

启动方式：
  python worker_pool.py --workers 4

Redis 模式（生产环境）：
  需要 Redis 服务运行在 localhost:6379
  python worker_pool.py --redis

本地模式（开发/演示，无需 Redis）：
  使用 multiprocessing.Queue 替代 Redis
  python worker_pool.py
"""

import json
import time
import uuid
import signal
import sys
import os
from multiprocessing import Process, Queue, cpu_count
from typing import Optional

# ==========================================
# Worker 进程
# ==========================================


def _worker_main(task_queue: Queue, result_queue: Queue, worker_id: int):
    """Worker 进程主循环：从队列取任务 → 执行 Agent → 写入结果。

    每个 Worker 是独立进程，拥有自己的 Python 解释器，
    绕过 GIL 限制，真正并行执行 CPU 密集型任务（如 ONNX 推理）。
    """
    print(f"[Worker-{worker_id}] 启动 (PID={os.getpid()})")

    # 在子进程中懒加载 Agent（避免 fork 时的资源竞争）
    from agent_async import AsyncAgentRunner
    import asyncio

    runner = AsyncAgentRunner()

    async def _process_task(task: dict) -> dict:
        """异步处理单个任务"""
        task_id = task["id"]
        query = task["query"]
        memory = task.get("memory", {})
        history = task.get("history", [])

        print(f"[Worker-{worker_id}] 处理任务 {task_id}: {query[:50]}...")

        final_state = None
        events = []
        async for event in runner.run(query, memory, history):
            events.append(event)
            if event["type"] == "done":
                final_state = event["final_state"]

        return {
            "task_id": task_id,
            "worker_id": worker_id,
            "final_state": final_state,
            "event_count": len(events),
        }

    while True:
        task = task_queue.get()
        if task is None:  # 停止信号
            print(f"[Worker-{worker_id}] 收到停止信号，退出")
            break

        try:
            result = asyncio.run(_process_task(task))
            result_queue.put(result)
        except Exception as e:
            print(f"[Worker-{worker_id}] 任务失败: {e}")
            result_queue.put({
                "task_id": task.get("id", "unknown"),
                "worker_id": worker_id,
                "error": str(e),
            })


# ==========================================
# 任务调度器（本地模式，无需 Redis）
# ==========================================


class LocalTaskBroker:
    """基于 multiprocessing.Queue 的本地任务调度器。

    适合开发环境，无需外部依赖。
    Redis 模式见下方 RedisTaskBroker。
    """

    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or cpu_count() * 2
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.workers = []
        self._results = {}  # task_id → result

    def start(self):
        """启动 Worker 进程池"""
        print(f"[Broker] 启动 {self.num_workers} 个 Worker 进程")
        for i in range(self.num_workers):
            p = Process(target=_worker_main, args=(self.task_queue, self.result_queue, i))
            p.daemon = True
            p.start()
            self.workers.append(p)
        print(f"[Broker] 全部 Worker 已就绪")

    def stop(self):
        """停止所有 Worker"""
        print("[Broker] 发送停止信号...")
        for _ in self.workers:
            self.task_queue.put(None)
        for p in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        print("[Broker] 全部 Worker 已停止")

    def submit(self, query: str, memory: dict = None, history: list = None) -> str:
        """提交任务到队列，返回 task_id"""
        task_id = str(uuid.uuid4())[:8]
        task = {
            "id": task_id,
            "query": query,
            "memory": memory or {},
            "history": history or [],
            "timestamp": time.time(),
        }
        self.task_queue.put(task)
        return task_id

    def get_result(self, task_id: str, timeout: float = 120) -> Optional[dict]:
        """阻塞等待任务结果"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if task_id in self._results:
                return self._results.pop(task_id)
            try:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                # 非阻塞检查结果队列
                if not self.result_queue.empty():
                    result = self.result_queue.get_nowait()
                    self._results[result["task_id"]] = result
                    if task_id == result["task_id"]:
                        return self._results.pop(task_id)
            except Exception:
                pass
            time.sleep(0.1)
        return None

    def collect_results(self, timeout: float = 5) -> list[dict]:
        """收集所有已完成任务的结果（非阻塞）"""
        results = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if not self.result_queue.empty():
                    results.append(self.result_queue.get_nowait())
                else:
                    break
            except Exception:
                break
            time.sleep(0.05)
        return results


# ==========================================
# Redis 任务调度器（生产环境）
# ==========================================


class RedisTaskBroker:
    """基于 Redis 的分布式任务调度器。

    支持多机部署：Worker 池可以分布在多台机器上，
    共享同一个 Redis 队列。

    需要: pip install redis
    Redis 启动: redis-server
    """

    def __init__(self, redis_host="localhost", redis_port=6379, queue_key="agent:tasks"):
        import redis
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.queue_key = queue_key
        self.result_prefix = "agent:results:"

    def submit(self, query: str, memory: dict = None, history: list = None) -> str:
        """提交任务到 Redis 队列"""
        task_id = str(uuid.uuid4())[:8]
        task = json.dumps({
            "id": task_id,
            "query": query,
            "memory": memory or {},
            "history": history or [],
            "timestamp": time.time(),
        }, ensure_ascii=False)
        self.redis.rpush(self.queue_key, task)
        return task_id

    def get_result(self, task_id: str, timeout: float = 120) -> Optional[dict]:
        """从 Redis 获取结果（阻塞等待）"""
        key = f"{self.result_prefix}{task_id}"
        result = self.redis.blpop(key, timeout=int(timeout))
        if result:
            return json.loads(result[1])
        return None

    def queue_length(self) -> int:
        """获取当前队列长度"""
        return self.redis.llen(self.queue_key)


# ==========================================
# 启动入口
# ==========================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent Worker Pool")
    parser.add_argument("--workers", type=int, default=cpu_count() * 2, help="Worker 进程数")
    parser.add_argument("--redis", action="store_true", help="使用 Redis 模式")
    parser.add_argument("--demo", action="store_true", help="运行演示")
    args = parser.parse_args()

    if args.demo:
        _run_demo(args.workers)
        return

    # 启动 Worker 池
    broker = LocalTaskBroker(num_workers=args.workers)
    broker.start()

    print(f"\n[Broker] Worker 池已启动，等待任务...")
    print(f"  模式: {'Redis' if args.redis else '本地 (multiprocessing.Queue)'}")
    print(f"  Worker 数: {args.workers}")
    print(f"  按 Ctrl+C 停止\n")

    # 等待中断信号
    def _shutdown(sig, frame):
        print("\n[Broker] 收到中断信号，正在停止...")
        broker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 保持主进程存活
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        broker.stop()


def _run_demo(num_workers: int):
    """演示：提交 3 个并发任务到 Worker 池"""
    print(f"=== Worker Pool 演示（{num_workers} Workers）===\n")

    broker = LocalTaskBroker(num_workers=num_workers)
    broker.start()

    # 提交任务
    queries = [
        "头痛发烧吃什么药",
        "糖尿病怎么预防",
        "高血压不能吃什么",
    ]

    task_ids = []
    for q in queries:
        tid = broker.submit(q)
        task_ids.append(tid)
        print(f"[Demo] 提交任务 {tid}: {q}")

    print(f"\n[Demo] 等待所有任务完成...")

    # 收集结果
    completed = set()
    deadline = time.time() + 180
    while len(completed) < len(task_ids) and time.time() < deadline:
        results = broker.collect_results(timeout=1)
        for r in results:
            tid = r.get("task_id", "")
            if tid in task_ids and tid not in completed:
                completed.add(tid)
                worker = r.get("worker_id", "?")
                events = r.get("event_count", 0)
                error = r.get("error", "")
                status = f"ERROR: {error}" if error else f"OK ({events} events)"
                print(f"[Demo] 任务 {tid} 完成 (Worker-{worker}): {status}")
        time.sleep(1)

    broker.stop()
    print(f"\n[Demo] 完成: {len(completed)}/{len(task_ids)}")


if __name__ == "__main__":
    main()
