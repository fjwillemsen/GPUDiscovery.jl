module GPUDiscoveryCUDAExt

using CUDA
using GPUDiscovery: GPUDiscovery, CUDAPlatform, DeviceInfo, _try, _int, format_pci, _check_index

function GPUDiscovery._package_devices(p::CUDAPlatform)
    CUDA.functional() || error("CUDA.jl is not functional on this system")
    attr(dev, a) = _int(_try(() -> CUDA.attribute(dev, a)))
    return [begin
        domain, bus, slot = attr(dev, CUDA.DEVICE_ATTRIBUTE_PCI_DOMAIN_ID),
                            attr(dev, CUDA.DEVICE_ATTRIBUTE_PCI_BUS_ID),
                            attr(dev, CUDA.DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
        DeviceInfo(;
            platform=p, index=i, name=CUDA.name(dev), source=:package,
            total_memory=_int(_try(() -> CUDA.totalmem(dev))),
            max_threads_per_block=attr(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
            compute_units=attr(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            capability=_try(() -> CUDA.capability(dev)),
            uuid=_try(() -> string(CUDA.uuid(dev))),
            pci_bus_id=any(isnothing, (domain, bus, slot)) ? nothing : format_pci(domain, bus, slot),
        )
    end for (i, dev) in enumerate(CUDA.devices())]
end

GPUDiscovery._ka_backend(::CUDAPlatform) = CUDA.CUDABackend()
GPUDiscovery._arraytype(::CUDAPlatform) = CUDA.CuArray

function GPUDiscovery._select_device!(p::CUDAPlatform, index::Int)
    _check_index(index, length(CUDA.devices()), p)
    CUDA.device!(index - 1)  # CUDA ordinals are 0-based
end

end
