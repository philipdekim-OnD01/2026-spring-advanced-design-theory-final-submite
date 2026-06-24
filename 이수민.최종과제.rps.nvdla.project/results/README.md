# Result files

- `native_strict_dla32_sweep_summary.csv`: accuracy and five-run latency summary
- `power_latency_summary.csv`: model/device power statistics
- `power_latency_comparison.csv`: GPU-DLA comparison and energy metrics
- `rps_gpu_dla_power_latency_sweep.xlsx`: formatted tables and charts
- `gpu_layer_info.json`, `dla_layer_info.json`: TensorRT layer information
- `trtexec_profile_gpu.txt`, `trtexec_profile_dla.txt`: representative TensorRT profiles

Raw `tegrastats`, Nsight and engine files are excluded because they are generated
artifacts or device-specific binaries.
