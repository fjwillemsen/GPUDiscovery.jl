"""
    DeviceInfo

Details about one device. Fields that could not be determined are `nothing`.

| field                   | meaning                                                                 |
|:------------------------|:------------------------------------------------------------------------|
| `platform`              | the [`Platform`](@ref) the device belongs to                            |
| `index`                 | 1-based position in `list_devices(platform)`, accepted by [`select_device!`](@ref) |
| `name`                  | marketing name, e.g. `"NVIDIA A100-SXM4-40GB"`                          |
| `total_memory`          | device memory in bytes (unified memory for Apple Silicon and the CPU)   |
| `max_threads_per_block` | maximum threads per block / workgroup / threadgroup                     |
| `compute_units`         | streaming multiprocessors, compute units, execution units, GPU cores or CPU threads |
| `capability`            | CUDA compute capability as a `VersionNumber`, AMD architecture (`"gfx90a"`), ... |
| `uuid`                  | device UUID as a string                                                 |
| `pci_bus_id`            | PCI address, e.g. `"0000:3b:00.0"`                                      |
| `source`                | `:package` when queried through the vendor package, `:system` otherwise |
"""
Base.@kwdef struct DeviceInfo
    platform::Platform
    index::Int
    name::String
    total_memory::Union{Int,Nothing} = nothing
    max_threads_per_block::Union{Int,Nothing} = nothing
    compute_units::Union{Int,Nothing} = nothing
    capability::Union{VersionNumber,String,Nothing} = nothing
    uuid::Union{String,Nothing} = nothing
    pci_bus_id::Union{String,Nothing} = nothing
    source::Symbol = :system
end

function Base.show(io::IO, d::DeviceInfo)
    print(io, "DeviceInfo(", d.platform, ", ", d.index, ", \"", d.name, "\")")
end

function Base.show(io::IO, ::MIME"text/plain", d::DeviceInfo)
    println(io, "DeviceInfo #", d.index, " on ", d.platform, " (via ", d.source, ")")
    print(io, "  name: ", d.name)
    for f in (:total_memory, :max_threads_per_block, :compute_units, :capability, :uuid, :pci_bus_id)
        v = getfield(d, f)
        v === nothing && continue
        print(io, "\n  ", f, ": ", f === :total_memory ? Base.format_bytes(v) : v)
    end
end

"Call `f()` and return `nothing` instead of throwing; used for best-effort property queries."
function _try(f)
    try
        return f()
    catch
        return nothing
    end
end

_int(x) = x === nothing ? nothing : Int(x)

"Format a PCI address as `dddd:bb:dd.f`."
function format_pci(domain::Integer, bus::Integer, device::Integer, func::Integer=0)
    hex(x, n) = string(x; base=16, pad=n)
    return "$(hex(domain, 4)):$(hex(bus, 2)):$(hex(device, 2)).$(string(func; base=16))"
end
