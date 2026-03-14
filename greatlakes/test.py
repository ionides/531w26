import os
import time
import csv
import jax
import jax.numpy as jnp

devices = jax.local_device_count()
print(f"JAX devices available: {devices}")
print(f"JAX default backend: {jax.default_backend()}")

key = jax.random.PRNGKey(0)

def time_task(fn):
    """Time a function, returning elapsed wall-clock seconds."""
    start = time.time()
    fn()
    elapsed = time.time() - start
    return elapsed

# Single large draw: 10^8 normal random values
def task0():
    x = jax.random.normal(key, shape=(10**8,))
    x.block_until_ready()

# 10 draws of 10^7
def task1():
    keys = jax.random.split(key, 10)
    xs = jax.vmap(lambda k: jax.random.normal(k, shape=(10**7,)))(keys)
    xs.block_until_ready()

# 10^2 draws of 10^6
def task2():
    keys = jax.random.split(key, 10**2)
    xs = jax.vmap(lambda k: jax.random.normal(k, shape=(10**6,)))(keys)
    xs.block_until_ready()

# 10^3 draws of 10^5
def task3():
    keys = jax.random.split(key, 10**3)
    xs = jax.vmap(lambda k: jax.random.normal(k, shape=(10**5,)))(keys)
    xs.block_until_ready()

# 10^4 draws of 10^4
def task4():
    keys = jax.random.split(key, 10**4)
    xs = jax.vmap(lambda k: jax.random.normal(k, shape=(10**4,)))(keys)
    xs.block_until_ready()

times = {}
for name, fn in [("time0", task0), ("time1", task1), ("time2", task2),
                 ("time3", task3), ("time4", task4)]:
    elapsed = time_task(fn)
    times[name] = elapsed
    print(f"{name}: {elapsed:.3f}s")

with open("test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["task", "elapsed"])
    for name, elapsed in times.items():
        writer.writerow([name, f"{elapsed:.4f}"])
