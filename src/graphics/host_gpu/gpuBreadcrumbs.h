#ifndef GRAPHICS_HOST_GPU_GPU_BREADCRUMBS_H_
#define GRAPHICS_HOST_GPU_GPU_BREADCRUMBS_H_

// Diagnostic: KYTY_GPU_BREADCRUMBS=1 (needs VK_AMD_buffer_marker) writes a marker before and after
// every guest draw and dispatch. When the device is lost, the markers show which shaders were
// still executing, so a GPU hang (TDR) can be traced to the guest work that caused it.

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics::GpuBreadcrumbs {

// Whether the user asked for breadcrumbs (the device extension is only added then).
bool Requested();
// After device creation, with VK_AMD_buffer_marker enabled.
void Init(vk::PhysicalDevice physical_device, vk::Device device);
bool Enabled();
// Returns the marker sequence to pass to End; 0 when disabled.
// info: free-form launch parameters (group counts, instances) printed in hex.
uint32_t Begin(vk::CommandBuffer command, const char* kind, uint64_t hash_a, uint64_t hash_b,
               uint64_t info = 0);
void     End(vk::CommandBuffer command, uint32_t sequence);
// Prints the work that had started but not finished.
void Report();

} // namespace Libs::Graphics::GpuBreadcrumbs

#endif // GRAPHICS_HOST_GPU_GPU_BREADCRUMBS_H_
