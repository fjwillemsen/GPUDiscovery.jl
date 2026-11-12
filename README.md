# GPUDiscovery.jl

Tired of manually adjusting which backend and array type your Julia kernels use when switching GPUs?
Use the light-weight `GPUDiscovery.jl` package to easily find out which GPU platform a machine has (NVIDIA/CUDA, AMD/ROCm, Intel/oneAPI or Apple/Metal) and query its devices through one interface. GPUDiscovery has no dependencies: CUDA.jl, AMDGPU.jl, oneAPI.jl, Metal.jl and KernelAbstractions.jl are optional, through package extensions.

```julia
using GPUDiscovery

platform = detect()          # CUDAPlatform(), ROCmPlatform(), oneAPIPlatform(), MetalPlatform() or CPUPlatform()
devices(platform)            # Vector{DeviceInfo}
```

```
julia> devices(detect())[1]
DeviceInfo #1 on MetalPlatform() (via system)
  name: Apple M5 Pro
  total_memory: 64.000 GiB
  compute_units: 20
```

## Detection

`detect()` returns the one GPU platform on the machine. It:

- throws an error if more than one GPU platform is found (`detect_all()` lists them all);
- returns `CPUPlatform()` if no GPU is found (`detect(fallback_to_cpu=false)` throws instead);
- uses the platform named in `ENV["GPUDISCOVERY_PLATFORM"]` (e.g. `"cuda"`), if set, without detecting.

Detection uses the operating system only, so it is fast and never loads a GPU package:

| platform | detected by |
|:--|:--|
| CUDA   | `nvidia-smi`, or an NVIDIA display device in Linux sysfs |
| ROCm   | an AMD display device in Linux sysfs plus `/dev/kfd`, or `rocm-smi` / `amd-smi` |
| oneAPI | an Intel display device in Linux sysfs, or `xpu-smi`. Only reported if there is no NVIDIA or AMD GPU, because integrated Intel GPUs sit next to most discrete GPUs |
| Metal  | macOS on Apple Silicon |

To choose a platform yourself: `GPUDiscovery.platform("amd")`.

## Device details

`devices(p)` returns a `DeviceInfo` per device, with these fields (`nothing` if unknown):
`platform`, `index` (1-based), `name`, `total_memory` (bytes), `max_threads_per_block`,
`compute_units`, `capability`, `uuid`, `pci_bus_id` and `source`.

Without the vendor package loaded, the details come from the operating system (`source = :system`).
Once the vendor package is loaded (`using CUDA`, or `GPUDiscovery.load(p)`), `devices` asks it
instead, which fills in more fields (`source = :package`). You can choose with the `query` keyword:

```julia
devices(p; query=:system)    # OS tools only, never loads a GPU package
devices(p; query=:package)   # load the vendor package and ask it
```

On Apple Silicon, `total_memory` is the physical (unified) memory from the system, and Metal's
recommended maximum working set size from the package.

## Running code on the detected platform

These functions load the vendor package when needed. It must be installed in the active
environment; GPUDiscovery never installs packages.

```julia
p = detect()
backend = ka_backend(p)      # CUDABackend(), ROCBackend(), oneAPIBackend(), MetalBackend() or CPU()
AT = arraytype(p)            # CuArray, ROCArray, oneArray, MtlArray or Array
select_device!(p, 2)         # 1-based for every platform, including CUDA
x = AT(rand(Float32, 1024))
```

Other helpers: `GPUDiscovery.package_name(p)`, `GPUDiscovery.vendor(p)`,
`GPUDiscovery.float_types(p)`, `GPUDiscovery.is_loaded(p)`.
