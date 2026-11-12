module GPUDiscoveryAMDGPUExt

using AMDGPU
using GPUDiscovery: GPUDiscovery, ROCmPlatform, DeviceInfo, _try, _int, format_pci, _check_index

function GPUDiscovery._package_devices(p::ROCmPlatform)
    AMDGPU.functional() || error("AMDGPU.jl is not functional on this system")
    return [begin
        props = _try(() -> AMDGPU.HIP.properties(dev))
        prop(f) = props === nothing ? nothing : _try(() -> getproperty(props, f))
        domain, bus, slot = prop(:pciDomainID), prop(:pciBusID), prop(:pciDeviceID)
        DeviceInfo(;
            platform=p, index=i, name=AMDGPU.HIP.name(dev), source=:package,
            total_memory=_int(prop(:totalGlobalMem)),
            max_threads_per_block=_int(prop(:maxThreadsPerBlock)),
            compute_units=_int(prop(:multiProcessorCount)),
            capability=_try(() -> AMDGPU.HIP.gcn_arch(dev)),
            pci_bus_id=any(isnothing, (domain, bus, slot)) ? nothing : format_pci(domain, bus, slot),
        )
    end for (i, dev) in enumerate(AMDGPU.devices())]
end

GPUDiscovery._ka_backend(::ROCmPlatform) = AMDGPU.ROCBackend()
GPUDiscovery._arraytype(::ROCmPlatform) = AMDGPU.ROCArray

function GPUDiscovery._select_device!(p::ROCmPlatform, index::Int)
    devs = AMDGPU.devices()
    _check_index(index, length(devs), p)
    AMDGPU.device!(devs[index])
end

end
