import jax
import jax.numpy as jnp
import time

# Initialize your initial key
key = jax.random.key(42)

# Define the function, then JIT-compile it with shape as a static argument.
# 'shape' must be static because jax.random.normal requires a concrete (non-traced)
# shape value at compile time. Passing static_argnames=['shape'] tells JAX to treat
# 'shape' as a compile-time constant rather than an abstract traced value.
# JAX will recompile the function if called with a different shape, which is fine here.
def get_random_normal(key, shape):
    new_key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape)
    return new_key, x

N = 10**8
get_random_normal_jit = jax.jit(get_random_normal, static_argnames=['shape'])

# Compare different ways to calculate N iid standard normal realizations
# block_until_ready is used because JAX does asynchronous dispatach
# https://docs.jax.dev/en/latest/async_dispatch.html

start_time = time.perf_counter()
key0, x0 = get_random_normal(key, shape=(N,))
x0.block_until_ready()
end0_time = time.perf_counter()
key1, x1 = get_random_normal_jit(key0, shape=(N,))
x1.block_until_ready()
end1_time = time.perf_counter()
key2, x2 = get_random_normal_jit(key1, shape=(N,))
x2.block_until_ready()
end2_time = time.perf_counter()
keys3 = jax.random.split(key2, 10)
keys3_new, x3 = jax.vmap(lambda k: get_random_normal_jit(k, shape=(int(N/10),)))(keys3)
x3.block_until_ready()
end3_time = time.perf_counter()

print(f"time0: {end0_time-start_time:.3f}s, ",
      f"time1: {end1_time-end0_time:.3f}s, ",
      f"time2: {end2_time-end1_time:.3f}s, ",
      f"time3: {end3_time-end2_time:.3f}s")

# This code was written in collaboration with Claude Code,
# using the opus 4.6 and sonnet 4.6 models and no
# user-supplied skills or context.
