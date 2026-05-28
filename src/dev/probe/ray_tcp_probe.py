"""Minimal Ray actor TCP probe.

Run:
    python src/dev/probe/ray_tcp_probe.py
    python src/dev/probe/ray_tcp_probe.py --num-workers 4 --use-gpu
"""

import argparse
import json
import os
import socket
import threading
import time


class TcpActor:
    def __init__(self, port=0):
        import ray

        self.host = ray.util.get_node_ip_address()
        self.messages = []
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", int(port)))
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        self.closed = False
        threading.Thread(target=self._serve, daemon=True).start()

    def endpoint(self):
        return {"host": self.host, "port": self.port}

    def wait_for_messages(self, count, timeout_sec=20):
        deadline = time.time() + timeout_sec
        while time.time() < deadline and len(self.messages) < count:
            time.sleep(0.05)
        return list(self.messages)

    def close(self):
        self.closed = True
        self.sock.close()

    def _serve(self):
        while not self.closed:
            try:
                conn, _ = self.sock.accept()
                data = conn.recv(65536)
                conn.close()
                self.messages.append(json.loads(data.decode("utf-8")))
            except OSError:
                break


def train_loop(config):
    import ray
    from ray import train

    endpoint = config["endpoint"]
    ctx = train.get_context()
    msg = {
        "rank": ctx.get_world_rank(),
        "local_rank": ctx.get_local_rank(),
        "world_size": ctx.get_world_size(),
        "pid": os.getpid(),
        "node_ip": ray.util.get_node_ip_address(),
    }

    with socket.create_connection((endpoint["host"], endpoint["port"])) as s:
        s.sendall(json.dumps(msg).encode("utf-8"))

    print(f"sent rank={msg['rank']} to {endpoint['host']}:{endpoint['port']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    import ray
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    ray.init(address=args.ray_address)

    Actor = ray.remote(num_cpus=1)(TcpActor)
    actor = Actor.remote(args.port)
    endpoint = ray.get(actor.endpoint.remote())
    print("actor endpoint:", endpoint)

    trainer = TorchTrainer(
        train_loop,
        train_loop_config={"endpoint": endpoint},
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=args.use_gpu,
        ),
    )
    trainer.fit()

    messages = ray.get(actor.wait_for_messages.remote(args.num_workers))
    ray.get(actor.close.remote())

    for msg in sorted(messages, key=lambda item: item.get("rank", 999)):
        print(msg)

    ranks = sorted(msg["rank"] for msg in messages if "rank" in msg)
    expected = list(range(args.num_workers))
    if ranks != expected:
        raise SystemExit(f"FAILED: expected ranks {expected}, got {ranks}")

    print("OK")


if __name__ == "__main__":
    main()
