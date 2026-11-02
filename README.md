# gpu-cost-meter

Find out how much your GPU inference costs and how much is wasted on idle time.

## Install (TestPyPI)

    pip install --index-url https://test.pypi.org/simple/ gpu-cost-meter

## Usage

    from gpu_cost_meter import meter, report

    @meter("my-endpoint")
    def predict(x):
        return model(x)

    for i in range(100):
        predict(i)

    report()

## Output

    === GPU Inference Cost Report ===
    Calls: 100
    Window: 3.2s
    GPU rate: $0.526/hr
    Window cost: $0.0005
    Cost per 1k inferences: $0.0047
    Idle cost (util<5%): $0.0001  <-- paid but wasted

    By endpoint:
      my-endpoint: 100 calls, avg GPU util 42.0%

## Notes

- `GPU_RATE` env var overrides the default ($0.526/hr, g4dn.xlarge on-demand).
- Requires `nvidia-smi` for GPU utilization. Without it, utilization shows `n/a`.
- Estimates are based on instance pricing, not your actual AWS bill.