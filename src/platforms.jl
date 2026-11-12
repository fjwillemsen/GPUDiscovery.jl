"""
    Platform

Abstract supertype of the compute platforms that GPUDiscovery knows about. Each platform is a
singleton type: [`CUDAPlatform`](@ref), [`ROCmPlatform`](@ref), [`oneAPIPlatform`](@ref),
[`MetalPlatform`](@ref) and [`CPUPlatform`](@ref).
"""
abstract type Platform end

"NVIDIA GPUs through CUDA.jl."
struct CUDAPlatform <: Platform end
"AMD GPUs through AMDGPU.jl (ROCm/HIP)."
struct ROCmPlatform <: Platform end
"Intel GPUs through oneAPI.jl (Level Zero)."
struct oneAPIPlatform <: Platform end
"Apple Silicon GPUs through Metal.jl."
struct MetalPlatform <: Platform end
"The host CPU, through KernelAbstractions' `CPU()` backend."
struct CPUPlatform <: Platform end

const GPU_PLATFORMS = (CUDAPlatform(), ROCmPlatform(), oneAPIPlatform(), MetalPlatform())
const ALL_PLATFORMS = (GPU_PLATFORMS..., CPUPlatform())

"""
    package_name(p::Platform) -> Union{String,Nothing}

Name of the Julia package that drives platform `p` (`nothing` for the CPU).
"""
package_name(::CUDAPlatform) = "CUDA"
package_name(::ROCmPlatform) = "AMDGPU"
package_name(::oneAPIPlatform) = "oneAPI"
package_name(::MetalPlatform) = "Metal"
package_name(::CPUPlatform) = nothing

package_uuid(::CUDAPlatform) = Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba")
package_uuid(::ROCmPlatform) = Base.UUID("21141c5a-9bdb-4563-92ae-f87d6854732e")
package_uuid(::oneAPIPlatform) = Base.UUID("8f75cd03-7ff8-4ecb-9b8f-daf728133b1b")
package_uuid(::MetalPlatform) = Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962")
package_uuid(::CPUPlatform) = Base.UUID("63c18a36-062a-441e-b654-da1e3ab1ce7c")  # KernelAbstractions

pkgid(p::Platform) = Base.PkgId(package_uuid(p), something(package_name(p), "KernelAbstractions"))

extension_name(::CUDAPlatform) = :GPUDiscoveryCUDAExt
extension_name(::ROCmPlatform) = :GPUDiscoveryAMDGPUExt
extension_name(::oneAPIPlatform) = :GPUDiscoveryoneAPIExt
extension_name(::MetalPlatform) = :GPUDiscoveryMetalExt
extension_name(::CPUPlatform) = :GPUDiscoveryKernelAbstractionsExt

"""
    vendor(p::Platform) -> String

Hardware vendor of the platform.
"""
vendor(::CUDAPlatform) = "NVIDIA"
vendor(::ROCmPlatform) = "AMD"
vendor(::oneAPIPlatform) = "Intel"
vendor(::MetalPlatform) = "Apple"
vendor(::CPUPlatform) = string(Sys.ARCH)

"""
    arraytype_name(p::Platform) -> Symbol

Name of the array type of the platform's package (e.g. `:CuArray`), usable without loading it.
See [`arraytype`](@ref) for the type itself.
"""
arraytype_name(::CUDAPlatform) = :CuArray
arraytype_name(::ROCmPlatform) = :ROCArray
arraytype_name(::oneAPIPlatform) = :oneArray
arraytype_name(::MetalPlatform) = :MtlArray
arraytype_name(::CPUPlatform) = :Array

"""
    ka_backend_name(p::Platform) -> Symbol

Name of the KernelAbstractions backend type of the platform (e.g. `:CUDABackend`).
"""
ka_backend_name(::CUDAPlatform) = :CUDABackend
ka_backend_name(::ROCmPlatform) = :ROCBackend
ka_backend_name(::oneAPIPlatform) = :oneAPIBackend
ka_backend_name(::MetalPlatform) = :MetalBackend
ka_backend_name(::CPUPlatform) = :CPU

"""
    float_types(p::Platform) -> Vector{Symbol}

Floating-point element types supported in kernels on this platform. `BFloat16` requires the
BFloat16s.jl package.
"""
float_types(::Platform) = [:Float16, :BFloat16, :Float32, :Float64]
float_types(::MetalPlatform) = [:Float16, :BFloat16, :Float32]  # no Float64 on Apple GPUs
float_types(::CPUPlatform) = [:Float16, :Float32, :Float64]

Base.show(io::IO, p::Platform) = print(io, nameof(typeof(p)), "()")

const PLATFORM_ALIASES = Dict(
    "cuda" => CUDAPlatform(), "nvidia" => CUDAPlatform(),
    "rocm" => ROCmPlatform(), "amd" => ROCmPlatform(), "amdgpu" => ROCmPlatform(), "hip" => ROCmPlatform(),
    "oneapi" => oneAPIPlatform(), "intel" => oneAPIPlatform(),
    "metal" => MetalPlatform(), "apple" => MetalPlatform(),
    "cpu" => CPUPlatform(),
)

"""
    platform(name::Union{AbstractString,Symbol}) -> Platform

Look up a platform by a case-insensitive name, e.g. `"cuda"`, `"nvidia"`, `"amd"`, `"rocm"`,
`"intel"`, `"oneapi"`, `"metal"`, `"apple"` or `"cpu"`.
"""
function platform(name::Union{AbstractString,Symbol})
    key = lowercase(strip(string(name)))
    haskey(PLATFORM_ALIASES, key) && return PLATFORM_ALIASES[key]
    throw(ArgumentError("Unknown platform \"$name\". Valid names: $(join(sort!(collect(keys(PLATFORM_ALIASES))), ", "))."))
end
