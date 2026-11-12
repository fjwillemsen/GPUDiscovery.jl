module GPUDiscoveryoneAPIExt

using oneAPI
using GPUDiscovery: GPUDiscovery, oneAPIPlatform, DeviceInfo, _try, _int, _check_index

_devices() = collect(oneAPI.devices(oneAPI.driver()))

function GPUDiscovery._package_devices(p::oneAPIPlatform)
    oneAPI.functional() || error("oneAPI.jl is not functional on this system")
    return [begin
        props = _try(() -> oneAPI.oneL0.properties(dev))
        prop(f) = props === nothing ? nothing : _try(() -> getproperty(props, f))
        DeviceInfo(;
            platform=p, index=i, name=something(prop(:name), "Intel GPU $i"), source=:package,
            total_memory=_int(_try(() -> sum(m -> m.totalSize, oneAPI.oneL0.memory_properties(dev)))),
            max_threads_per_block=_int(_try(() -> oneAPI.oneL0.compute_properties(dev).maxTotalGroupSize)),
            compute_units=_try(() -> Int(props.numSlices * props.numSubslicesPerSlice * props.numEUsPerSubslice)),
            uuid=_try(() -> string(props.uuid)),
        )
    end for (i, dev) in enumerate(_devices())]
end

GPUDiscovery._ka_backend(::oneAPIPlatform) = oneAPI.oneAPIBackend()
GPUDiscovery._arraytype(::oneAPIPlatform) = oneAPI.oneArray

function GPUDiscovery._select_device!(p::oneAPIPlatform, index::Int)
    devs = _devices()
    _check_index(index, length(devs), p)
    oneAPI.device!(devs[index])
end

end
