using GPUDiscovery
using GPUDiscovery: platform, package_name, is_loaded, system_devices, format_pci, _normalize_pci
using Test
using Aqua

@testset "GPUDiscovery" begin
    @testset "Aqua" begin
        Aqua.test_all(GPUDiscovery)
    end

    @testset "platform lookup" begin
        @test platform("CUDA") === CUDAPlatform()
        @test platform(:amd) === ROCmPlatform()
        @test platform(" Intel ") === oneAPIPlatform()
        @test platform("apple") === MetalPlatform()
        @test platform("cpu") === CPUPlatform()
        @test_throws ArgumentError platform("tpu")
        @test package_name(CUDAPlatform()) == "CUDA"
        @test package_name(CPUPlatform()) === nothing
    end

    @testset "PCI formatting" begin
        @test format_pci(0, 0x3b, 0) == "0000:3b:00.0"
        @test _normalize_pci("00000000:3B:00.0") == "0000:3b:00.0"
    end

    @testset "detection" begin
        found = detect_all()
        @test found isa Vector{GPUDiscovery.Platform}
        @test CPUPlatform() ∉ found
        if length(found) <= 1
            p = detect()
            @test p === (isempty(found) ? CPUPlatform() : only(found))
        else
            @test_throws ErrorException detect()
        end
        isempty(found) && @test_throws ErrorException detect(fallback_to_cpu=false)

        withenv(GPUDiscovery.OVERRIDE_ENV => "cpu") do
            @test detect() === CPUPlatform()
        end
    end

    @testset "devices (system)" begin
        cpu = only(devices(CPUPlatform()))
        @test cpu.index == 1
        @test cpu.compute_units == Sys.CPU_THREADS
        @test cpu.total_memory > 0
        for p in detect_all()
            devs = devices(p; query=:system)
            @test !isempty(devs)
            @test all(d -> d.platform === p && d.source === :system, devs)
            @test [d.index for d in devs] == 1:length(devs)
        end
        @test arraytype(CPUPlatform()) === Array
        @test sprint(show, MIME("text/plain"), cpu) isa String
    end

    # Package-level queries; only run when the vendor package is installed in the test environment.
    @testset "devices (package)" begin
        for p in detect_all()
            Base.find_package(package_name(p)) === nothing && continue
            devs = devices(p; query=:package)
            @test is_loaded(p)
            @test all(d -> d.source === :package, devs)
            @test all(d -> d.max_threads_per_block !== nothing && d.max_threads_per_block > 0, devs)
            @test nameof(typeof(ka_backend(p))) === GPUDiscovery.ka_backend_name(p)
            @test nameof(arraytype(p)) === GPUDiscovery.arraytype_name(p)
            select_device!(p, 1)
            @test_throws ArgumentError select_device!(p, length(devs) + 1)
        end
    end
end
