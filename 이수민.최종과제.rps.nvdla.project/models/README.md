# Model download

TensorRT engine files are intentionally excluded because they are tied to the
Jetson/TensorRT environment. `download_models.sh` downloads the portable ONNX
and full-integer INT8 TFLite artifacts from this fork's GitHub Release.

```bash
./download_models.sh all
```

Available modes:

```text
onnx    ONNX model, manifest and model metadata
tflite  Full-integer INT8 TFLite model
all     Both archives
```

Release artifacts:

| File | Size | SHA-256 |
| --- | ---: | --- |
| `rps_nvdla_onnx_models_v1.tar.gz` | 295 MiB | `82f07e065e1e9eb75fdf0600aa0c80a0828d1844f43e7915748c3ea9c2876d33` |
| `rps_nvdla_tflite_models_v1.tar.gz` | 76 MiB | `bd3a56f6e5e5d240fdbc0d5aeca73d2c1a37b3852f85ac5e8d79e4922dfa1f1a` |

Both archives contain the seven width-sweep variants (`w0_25` through
`w16_0`). The ONNX archive also includes each model's metadata and manifest.
