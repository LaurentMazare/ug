import numpy as np
from tinygrad import Tensor, Device
import time

N_ROWS = 65536
N_STEPS = 100

_device = Device[Device.DEFAULT]

for N_COLS in [128, 256, 512, 1024, 2048, 4096]:
    data = np.random.randn(N_ROWS, N_COLS).astype(np.float32)
    arg = Tensor(data).realize()
    for _ in range(3):
        _foo = arg.softmax().realize()
        _device.synchronize()
    
    start_time = time.time()
    for _ in range(N_STEPS):
        _foo = arg.softmax().realize()
        _device.synchronize()
    dt = time.time() - start_time
    avg_time = dt / N_STEPS
    gb_per_s = 8e-9 * N_ROWS * N_COLS * N_STEPS / dt
    print(f"{N_COLS:5}   time: {avg_time * 1000:6.2f}ms    GB/s: {gb_per_s:.0f}")

