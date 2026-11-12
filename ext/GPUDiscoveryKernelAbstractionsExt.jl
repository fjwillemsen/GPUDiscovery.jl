module GPUDiscoveryKernelAbstractionsExt

using KernelAbstractions
using GPUDiscovery: GPUDiscovery, CPUPlatform

GPUDiscovery._ka_backend(::CPUPlatform) = KernelAbstractions.CPU()

end
