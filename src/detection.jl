const OVERRIDE_ENV = "GPUDISCOVERY_PLATFORM"

"""
    detect_all() -> Vector{Platform}

All GPU platforms for which the operating system reports hardware. The CPU is not included.

Intel GPUs are only reported when no NVIDIA or AMD GPU is present, because most machines with a
discrete GPU also have an Intel integrated GPU, which is rarely the one you want to compute on.
"""
function detect_all()
    found = Platform[p for p in (CUDAPlatform(), ROCmPlatform(), MetalPlatform()) if system_available(p)]
    if isempty(found) && system_available(oneAPIPlatform())
        push!(found, oneAPIPlatform())
    end
    return found
end

"""
    detect(; fallback_to_cpu=true) -> Platform

The single GPU platform available on this machine.

- Throws an error if more than one GPU platform is found; pick one with
  `GPUDiscovery.platform("cuda")` or set the environment variable `$OVERRIDE_ENV`.
- Returns [`CPUPlatform`](@ref) if no GPU is found, or throws if `fallback_to_cpu=false`.

Setting `ENV["$OVERRIDE_ENV"]` (e.g. to `"cuda"`, `"amd"`, `"intel"`, `"metal"` or `"cpu"`)
skips detection.
"""
function detect(; fallback_to_cpu::Bool=true)
    forced = get(ENV, OVERRIDE_ENV, "")
    isempty(forced) || return platform(forced)

    found = detect_all()
    length(found) == 1 && return only(found)
    if length(found) > 1
        error("Multiple GPU platforms detected: $(join(found, ", ")). Choose one explicitly, e.g. " *
              "`GPUDiscovery.platform(\"$(lowercase(package_name(first(found))))\")`, or set " *
              "ENV[\"$OVERRIDE_ENV\"].")
    end
    fallback_to_cpu && return CPUPlatform()
    error("No GPU detected. Use `detect(fallback_to_cpu=true)` to fall back to the CPU.")
end
