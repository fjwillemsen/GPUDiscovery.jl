"""
    GPUDiscovery

Detect which GPU platform (CUDA, ROCm, oneAPI, Metal) is available on this machine and query
its devices through one interface, without depending on any GPU package.

```julia
using GPUDiscovery
platform = detect()          # e.g. CUDAPlatform(); errors if several GPU platforms are present
list_devices(platform)            # Vector{DeviceInfo}
```

Detection only uses the operating system (sysfs, `nvidia-smi`, ...). Richer device details, the
KernelAbstractions backend and the array type become available once the vendor package
(CUDA.jl, AMDGPU.jl, oneAPI.jl or Metal.jl) is loaded, either by `using` it yourself or with
[`GPUDiscovery.load`](@ref). This works through package extensions, so those packages are never
hard dependencies.
"""
module GPUDiscovery

export detect, detect_all, list_devices, DeviceInfo, ka_backend, arraytype, select_device!
export CUDAPlatform, ROCmPlatform, oneAPIPlatform, MetalPlatform, CPUPlatform

include("platforms.jl")
include("deviceinfo.jl")
include("system.jl")
include("detection.jl")
include("interface.jl")

end # module
