#include "graphics/host_gpu/gpuBreadcrumbs.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::GpuBreadcrumbs {

namespace {

struct Record {
	std::atomic<uint32_t> sequence {0};
	const char*           kind   = nullptr;
	uint64_t              hash_a = 0;
	uint64_t              hash_b = 0;
};

constexpr uint32_t                  kRingSize = 1u << 16u;
std::array<Record, kRingSize>       g_ring;
std::atomic<uint32_t>               g_next {0};
vk::Buffer                          g_buffer  = nullptr;
vk::DeviceMemory                    g_memory  = nullptr;
volatile uint32_t*                  g_markers = nullptr; // [0] begun (top), [1] finished (bottom)
bool                                g_enabled = false;
PFN_vkCmdWriteBufferMarkerAMD       g_write   = nullptr;

} // namespace

bool Requested() {
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_GPU_BREADCRUMBS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return requested;
}

void Init(vk::PhysicalDevice physical_device, vk::Device device) {
	g_write = reinterpret_cast<PFN_vkCmdWriteBufferMarkerAMD>(
	    device.getProcAddr("vkCmdWriteBufferMarkerAMD"));
	if (g_write == nullptr) {
		std::printf("GPU breadcrumbs: vkCmdWriteBufferMarkerAMD unavailable\n");
		return;
	}
	vk::BufferCreateInfo buffer_info {};
	buffer_info.size        = 64;
	buffer_info.usage       = vk::BufferUsageFlagBits::eTransferDst;
	buffer_info.sharingMode = vk::SharingMode::eExclusive;
	if (device.createBuffer(&buffer_info, nullptr, &g_buffer) != vk::Result::eSuccess) return;
	vk::MemoryRequirements requirements {};
	device.getBufferMemoryRequirements(g_buffer, &requirements);
	const auto properties = physical_device.getMemoryProperties();
	const auto wanted = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
	uint32_t type = UINT32_MAX;
	for (uint32_t i = 0; i < properties.memoryTypeCount; i++) {
		if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
		    (properties.memoryTypes[i].propertyFlags & wanted) == wanted &&
		    !(properties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
			type = i;
			break;
		}
	}
	if (type == UINT32_MAX) return;
	vk::MemoryAllocateInfo allocate {};
	allocate.allocationSize  = requirements.size;
	allocate.memoryTypeIndex = type;
	if (device.allocateMemory(&allocate, nullptr, &g_memory) != vk::Result::eSuccess) return;
	if (device.bindBufferMemory(g_buffer, g_memory, 0) != vk::Result::eSuccess) return;
	void* mapped = nullptr;
	if (device.mapMemory(g_memory, 0, 64, {}, &mapped) != vk::Result::eSuccess) return;
	g_markers = static_cast<volatile uint32_t*>(mapped);
	g_markers[0] = 0;
	g_markers[1] = 0;
	g_enabled    = true;
	std::printf("GPU breadcrumbs: enabled\n");
}

bool Enabled() {
	return g_enabled;
}

uint32_t Begin(vk::CommandBuffer command, const char* kind, uint64_t hash_a, uint64_t hash_b) {
	if (!g_enabled) return 0;
	uint32_t sequence = g_next.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (sequence == 0) sequence = g_next.fetch_add(1, std::memory_order_relaxed) + 1u;
	auto& record = g_ring[sequence % kRingSize];
	record.kind   = kind;
	record.hash_a = hash_a;
	record.hash_b = hash_b;
	record.sequence.store(sequence, std::memory_order_release);
	g_write(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_buffer, 0, sequence);
	return sequence;
}

void End(vk::CommandBuffer command, uint32_t sequence) {
	if (!g_enabled || sequence == 0) return;
	g_write(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_buffer, 4, sequence);
}

void Report() {
	if (!g_enabled) {
		std::printf("GPU breadcrumbs: off (set KYTY_GPU_BREADCRUMBS=1)\n");
		std::fflush(stdout);
		return;
	}
	const uint32_t begun    = g_markers[0];
	const uint32_t finished = g_markers[1];
	const uint32_t recorded = g_next.load();
	std::printf("--- GPU breadcrumbs: recorded=%u last started=%u last finished=%u ---\n", recorded,
	            begun, finished);
	const auto print = [](uint32_t sequence, const char* note) {
		const auto& record = g_ring[sequence % kRingSize];
		if (record.sequence.load(std::memory_order_acquire) != sequence) return;
		std::printf("  %s #%u %s 0x%016" PRIx64 " 0x%016" PRIx64 "\n", note, sequence,
		            record.kind != nullptr ? record.kind : "?", record.hash_a, record.hash_b);
	};
	for (uint32_t s = finished > 4 ? finished - 4 : 1; s <= finished && s != 0; s++) print(s, "done   ");
	for (uint32_t s = finished + 1; s <= finished + 12 && s <= recorded; s++) {
		print(s, s <= begun ? "RUNNING" : "queued ");
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics::GpuBreadcrumbs
