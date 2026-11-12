# Functions that the package extensions add methods to. Always call them through
# `Base.invokelatest`, since the extension may have been loaded after the caller was compiled.
function _package_devices end
function _ka_backend end
function _arraytype end
function _select_device! end

"""
    is_loaded(p::Platform) -> Bool

Whether the vendor package of `p` (KernelAbstractions for the CPU) is loaded, so that
package-level queries are available.
"""
is_loaded(p::Platform) = Base.get_extension(@__MODULE__, extension_name(p)) !== nothing

"""
    GPUDiscovery.load(p::Platform) -> Module

Load the vendor package of `p` (CUDA.jl, AMDGPU.jl, oneAPI.jl, Metal.jl, or KernelAbstractions.jl
for the CPU), equivalent to `using CUDA` etc. The package must be installed in the active
environment; it is never installed automatically.
"""
function load(p::Platform)
    id = pkgid(p)
    mod = try
        Base.require(id)
    catch e
        e isa ArgumentError && error("$(id.name).jl is required for $p but is not installed in the active " *
                                     "environment. Install it with `import Pkg; Pkg.add(\"$(id.name)\")`.")
        rethrow()
    end
    is_loaded(p) || error("$(id.name) was loaded, but GPUDiscovery's extension for it did not activate. " *
                          "Check that the installed $(id.name) version is compatible with GPUDiscovery.")
    return mod
end

function _ensure_loaded(p::Platform, autoload::Bool, what)
    is_loaded(p) && return
    autoload && return (load(p); nothing)
    error("$what for $p needs $(pkgid(p).name).jl: run `using $(pkgid(p).name)` or `GPUDiscovery.load($p)` first.")
end

"""
    devices(p::Platform=detect(); query=:auto) -> Vector{DeviceInfo}

List the devices of platform `p`.

`query` selects where the information comes from:
- `:auto` (default): from the vendor package if it is loaded, otherwise from the operating system;
- `:package`: load the vendor package (see [`GPUDiscovery.load`](@ref)) and query it, which gives
  the most complete [`DeviceInfo`](@ref);
- `:system`: only use operating-system tools, never touching the vendor package.
"""
function devices(p::Platform=detect(); query::Symbol=:auto)
    query in (:auto, :package, :system) || throw(ArgumentError("`query` must be :auto, :package or :system, got :$query"))
    (p isa CPUPlatform || query === :system) && return system_devices(p)
    query === :package && load(p)
    is_loaded(p) || return system_devices(p)
    try
        return Base.invokelatest(_package_devices, p)::Vector{DeviceInfo}
    catch e
        query === :package && rethrow()
        @warn "Querying $p devices through $(package_name(p)).jl failed; falling back to system tools." exception = e
        return system_devices(p)
    end
end

"""
    ka_backend(p::Platform=detect(); autoload=true)

The KernelAbstractions backend instance for `p`, e.g. `CUDABackend()` or `CPU()`.
With `autoload=true` the vendor package is loaded if needed (see [`GPUDiscovery.load`](@ref)).
"""
function ka_backend(p::Platform=detect(); autoload::Bool=true)
    _ensure_loaded(p, autoload, "The KernelAbstractions backend")
    return Base.invokelatest(_ka_backend, p)
end

"""
    arraytype(p::Platform=detect(); autoload=true) -> Type

The array type of `p` (`CuArray`, `ROCArray`, `oneArray`, `MtlArray` or `Array`), which can be
used to move data to the device: `arraytype(p)(host_array)`.
"""
arraytype(::CPUPlatform; autoload::Bool=true) = Array
function arraytype(p::Platform=detect(); autoload::Bool=true)
    _ensure_loaded(p, autoload, "The array type")
    return Base.invokelatest(_arraytype, p)
end

"""
    select_device!(p::Platform, index::Integer; autoload=true)

Make device number `index` (1-based, as in [`devices`](@ref)) the active device for the current
task. A no-op for the CPU.
"""
select_device!(::CPUPlatform, index::Integer; autoload::Bool=true) =
    index == 1 ? nothing : throw(ArgumentError("The CPU platform has a single device, got index $index"))
function select_device!(p::Platform, index::Integer; autoload::Bool=true)
    _ensure_loaded(p, autoload, "Device selection")
    Base.invokelatest(_select_device!, p, Int(index))
    return nothing
end

function _check_index(index, n, p)
    1 <= index <= n || throw(ArgumentError("Device index $index out of range for $p, which has $n device(s)"))
end
