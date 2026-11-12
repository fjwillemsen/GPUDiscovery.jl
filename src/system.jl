# Operating-system level probes. Nothing here loads a GPU package, so it is cheap and safe to call.

"Run `cmd` and return its stdout, or `nothing` if the program is missing or fails."
function _readcmd(cmd::Cmd)
    out = try
        read(pipeline(cmd; stderr=devnull), String)  # throws if missing or non-zero exit
    catch
        return nothing
    end
    return isempty(strip(out)) ? nothing : out
end

_readfile(path) = isfile(path) ? _try(() -> strip(read(path, String))) : nothing

const PCI_VENDOR = Dict("0x10de" => CUDAPlatform(), "0x1002" => ROCmPlatform(), "0x8086" => oneAPIPlatform())

"""
    SysfsGPU

A display-class PCI device found under `/sys/class/drm` (Linux only).
"""
struct SysfsGPU
    platform::Platform
    path::String        # resolved sysfs device directory
    pci_bus_id::String
end

"List display-class PCI devices of known GPU vendors from Linux sysfs, without duplicates."
function sysfs_gpus()
    Sys.islinux() || return SysfsGPU[]
    drm = "/sys/class/drm"
    isdir(drm) || return SysfsGPU[]
    found = SysfsGPU[]
    seen = Set{String}()
    for entry in something(_try(() -> readdir(drm)), String[])
        occursin(r"^card\d+$", entry) || continue
        dev = _try(() -> realpath(joinpath(drm, entry, "device")))
        (dev === nothing || dev in seen) && continue
        push!(seen, dev)
        class = _readfile(joinpath(dev, "class"))
        (class === nothing || !startswith(class, "0x03")) && continue  # 0x03xxxx = display controller
        p = get(PCI_VENDOR, something(_readfile(joinpath(dev, "vendor")), ""), nothing)
        p === nothing && continue
        push!(found, SysfsGPU(p, dev, basename(dev)))
    end
    sort!(found; by=g -> g.pci_bus_id)
    return found
end

sysfs_gpus(p::Platform) = filter(g -> g.platform == p, sysfs_gpus())

"""
    nvidia_smi_devices() -> Union{Vector{Dict{String,String}},Nothing}

Query NVIDIA GPUs with `nvidia-smi`; `nothing` if it is unavailable.
"""
function nvidia_smi_devices()
    fields = ["name", "uuid", "pci.bus_id", "memory.total", "compute_cap"]
    out = _readcmd(`nvidia-smi --query-gpu=$(join(fields, ',')) --format=csv,noheader,nounits`)
    if out === nothing  # older drivers do not know compute_cap
        pop!(fields)
        out = _readcmd(`nvidia-smi --query-gpu=$(join(fields, ',')) --format=csv,noheader,nounits`)
    end
    out === nothing && return nothing
    rows = Dict{String,String}[]
    for line in eachsplit(strip(out), '\n')
        vals = strip.(split(line, ','))
        length(vals) == length(fields) || continue
        push!(rows, Dict(zip(fields, vals)))
    end
    return rows
end

"True on macOS with an Apple Silicon GPU (all of which support Metal 3, required by Metal.jl)."
apple_silicon() = Sys.isapple() && Sys.ARCH === :aarch64

_sysctl(key) = (s = _readcmd(`sysctl -n $key`); s === nothing ? nothing : strip(s))

# ---------------------------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------------------------

"""
    system_available(p::Platform) -> Bool

Whether the operating system reports hardware for platform `p`. Does not load any GPU package.
"""
system_available(::CPUPlatform) = true
system_available(::MetalPlatform) = apple_silicon()

function system_available(p::CUDAPlatform)
    rows = nvidia_smi_devices()
    rows !== nothing && return !isempty(rows)
    return !isempty(sysfs_gpus(p))
end

function system_available(p::ROCmPlatform)
    # The ROCm kernel driver exposes /dev/kfd; also accept a working rocm-smi / amd-smi.
    Sys.islinux() && isempty(sysfs_gpus(p)) && return false
    ispath("/dev/kfd") && return true
    return _readcmd(`rocm-smi`) !== nothing || _readcmd(`amd-smi list`) !== nothing
end

system_available(p::oneAPIPlatform) = !isempty(sysfs_gpus(p)) || _readcmd(`xpu-smi discovery`) !== nothing

# ---------------------------------------------------------------------------------------------
# Device listings from the operating system (used when the vendor package is not loaded)
# ---------------------------------------------------------------------------------------------

function _parse_cc(s)
    s === nothing && return nothing
    v = tryparse(VersionNumber, s)
    return v
end

function system_devices(p::CUDAPlatform)
    rows = nvidia_smi_devices()
    if rows !== nothing
        return [DeviceInfo(;
            platform=p, index=i, name=r["name"],
            total_memory=(m = tryparse(Int, r["memory.total"]); m === nothing ? nothing : m * 1024^2),  # MiB
            capability=_parse_cc(get(r, "compute_cap", nothing)),
            uuid=replace(r["uuid"], r"^GPU-" => ""),
            pci_bus_id=_normalize_pci(r["pci.bus_id"]),
        ) for (i, r) in enumerate(rows)]
    end
    return _sysfs_devices(p, "NVIDIA GPU")
end

system_devices(p::ROCmPlatform) = _sysfs_devices(p, "AMD GPU")
system_devices(p::oneAPIPlatform) = _sysfs_devices(p, "Intel GPU")

function system_devices(p::MetalPlatform)
    apple_silicon() || return DeviceInfo[]
    name = something(_sysctl("machdep.cpu.brand_string"), "Apple GPU")
    cores = _metal_gpu_cores()
    return [DeviceInfo(; platform=p, index=1, name, total_memory=Int(Sys.total_memory()), compute_units=cores)]
end

function system_devices(p::CPUPlatform)
    info = Sys.cpu_info()
    name = isempty(info) ? string(Sys.ARCH) : strip(first(info).model)
    return [DeviceInfo(; platform=p, index=1, name, total_memory=Int(Sys.total_memory()),
                       compute_units=Sys.CPU_THREADS)]
end

"Number of GPU cores on Apple Silicon, from the IORegistry."
function _metal_gpu_cores()
    out = _readcmd(`ioreg -rc AGXAccelerator -d1`)
    out === nothing && return nothing
    m = match(r"\"gpu-core-count\"\s*=\s*(\d+)", out)
    return m === nothing ? nothing : parse(Int, m[1])
end

"nvidia-smi reports `00000000:3B:00.0`; normalise to the 4-digit lowercase sysfs form."
function _normalize_pci(s)
    m = match(r"^([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-7])$", strip(s))
    m === nothing && return s
    return format_pci(parse(Int, m[1]; base=16), parse(Int, m[2]; base=16), parse(Int, m[3]; base=16), parse(Int, m[4]))
end

function _sysfs_devices(p::Platform, fallback_name)
    return [DeviceInfo(;
        platform=p, index=i,
        name=something(_readfile(joinpath(g.path, "product_name")), "$fallback_name ($(g.pci_bus_id))"),
        total_memory=_int(_try(() -> parse(Int, _readfile(joinpath(g.path, "mem_info_vram_total"))))),
        pci_bus_id=g.pci_bus_id,
    ) for (i, g) in enumerate(sysfs_gpus(p))]
end
