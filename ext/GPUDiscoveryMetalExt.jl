module GPUDiscoveryMetalExt

using Metal
using GPUDiscovery: GPUDiscovery, MetalPlatform, DeviceInfo, _try, _int, _check_index, _metal_gpu_cores

function GPUDiscovery._package_devices(p::MetalPlatform)
    Metal.functional() || error("Metal.jl is not functional on this system")
    return [DeviceInfo(;
        platform=p, index=i, name=String(dev.name), source=:package,
        total_memory=_int(_try(() -> dev.recommendedMaxWorkingSetSize)),
        max_threads_per_block=_int(_try(() -> dev.maxThreadsPerThreadgroup.width)),
        compute_units=_metal_gpu_cores(),
        uuid=_try(() -> string(dev.registryID; base=16)),
    ) for (i, dev) in enumerate(Metal.devices())]
end

GPUDiscovery._ka_backend(::MetalPlatform) = Metal.MetalBackend()
GPUDiscovery._arraytype(::MetalPlatform) = Metal.MtlArray

function GPUDiscovery._select_device!(p::MetalPlatform, index::Int)
    devs = Metal.devices()
    _check_index(index, length(devs), p)
    Metal.device!(devs[index])
end

end
