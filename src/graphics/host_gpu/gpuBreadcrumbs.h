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

// Indirect (task-path) mesh draws: what the emulator passed to the task shader.
struct MeshCapture {
	uint64_t guest_args     = 0; // guest address of the indirect records
	uint64_t records        = 0; // device address the task shader reads them from
	uint64_t index_address  = 0;
	uint32_t stride         = 0;
	uint32_t index_size     = 0;
	uint32_t index_limit    = 0;
	uint32_t primitive_size = 0;
	uint32_t group_step     = 0;
	uint32_t max_total      = 0;
};
// Reserves a capture slot for one indirect mesh draw. Returns the device address the task shader
// writes the record it read to (0 when disabled); *tag identifies the slot's use and goes into the
// breadcrumb info with MeshCaptureInfo.
uint64_t ReserveMeshCapture(const MeshCapture& cpu, uint32_t* tag);
// Breadcrumb info of a task-path mesh draw: 1<<63 | tag<<32 | record count.
inline uint64_t MeshCaptureInfo(uint32_t tag, uint32_t record_count) {
	return (uint64_t {1} << 63u) | (uint64_t {tag & 0x7fffffffu} << 32u) | record_count;
}

} // namespace Libs::Graphics::GpuBreadcrumbs

#endif // GRAPHICS_HOST_GPU_GPU_BREADCRUMBS_H_
