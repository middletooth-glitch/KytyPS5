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
	uint64_t              info   = 0;
};

constexpr uint32_t                  kRingSize = 1u << 16u;
std::array<Record, kRingSize>       g_ring;
std::atomic<uint32_t>               g_next {0};
vk::Buffer                          g_buffer  = nullptr;
vk::DeviceMemory                    g_memory  = nullptr;
volatile uint32_t*                  g_markers = nullptr; // [0] begun (top), [1] finished (bottom)
bool                                g_enabled = false;
PFN_vkCmdWriteBufferMarkerAMD       g_write   = nullptr;

// Mesh capture: per slot 8 dwords the task shader writes (tag, the five record dwords it read,
// its group count before the launch caps, and the group count it launched).
constexpr uint32_t                          kCaptureSlots     = 4096;
constexpr uint32_t                          kCaptureSlotDwords = 8;
vk::Buffer                                  g_capture_buffer  = nullptr;
vk::DeviceMemory                            g_capture_memory  = nullptr;
volatile uint32_t*                          g_capture         = nullptr;
uint64_t                                    g_capture_address = 0;
std::atomic<uint32_t>                       g_capture_next {0};
std::array<MeshCapture, kCaptureSlots>      g_capture_cpu;
std::array<std::atomic<uint32_t>, kCaptureSlots> g_capture_cpu_tag {};

uint32_t FindHostMemoryType(vk::PhysicalDevice physical_device, uint32_t type_bits) {
	const auto properties = physical_device.getMemoryProperties();
	const auto wanted = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
	for (uint32_t i = 0; i < properties.memoryTypeCount; i++) {
		if ((type_bits & (1u << i)) != 0 &&
		    (properties.memoryTypes[i].propertyFlags & wanted) == wanted &&
		    !(properties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
			return i;
		}
	}
	return UINT32_MAX;
}

void InitMeshCapture(vk::PhysicalDevice physical_device, vk::Device device) {
	constexpr vk::DeviceSize size = vk::DeviceSize {kCaptureSlots} * kCaptureSlotDwords * 4u;
	vk::BufferCreateInfo buffer_info {};
	buffer_info.size        = size;
	buffer_info.usage       = vk::BufferUsageFlagBits::eStorageBuffer |
	                          vk::BufferUsageFlagBits::eShaderDeviceAddress;
	buffer_info.sharingMode = vk::SharingMode::eExclusive;
	if (device.createBuffer(&buffer_info, nullptr, &g_capture_buffer) != vk::Result::eSuccess) return;
	vk::MemoryRequirements requirements {};
	device.getBufferMemoryRequirements(g_capture_buffer, &requirements);
	const uint32_t type = FindHostMemoryType(physical_device, requirements.memoryTypeBits);
	if (type == UINT32_MAX) return;
	vk::MemoryAllocateFlagsInfo flags {};
	flags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
	vk::MemoryAllocateInfo allocate {};
	allocate.pNext           = &flags;
	allocate.allocationSize  = requirements.size;
	allocate.memoryTypeIndex = type;
	if (device.allocateMemory(&allocate, nullptr, &g_capture_memory) != vk::Result::eSuccess) return;
	static_cast<void>(device.bindBufferMemory(g_capture_buffer, g_capture_memory, 0));
	void* mapped = nullptr;
	if (device.mapMemory(g_capture_memory, 0, size, {}, &mapped) != vk::Result::eSuccess) return;
	std::memset(mapped, 0, static_cast<size_t>(size));
	vk::BufferDeviceAddressInfo address_info {};
	address_info.buffer = g_capture_buffer;
	g_capture_address   = device.getBufferAddress(address_info);
	if (g_capture_address == 0) return;
	g_capture = static_cast<volatile uint32_t*>(mapped);
	std::printf("GPU breadcrumbs: indirect mesh draw capture enabled\n");
}

void PrintMeshCapture(uint32_t tag) {
	if (g_capture == nullptr || tag == 0) return;
	const uint32_t slot = tag % kCaptureSlots;
	if (g_capture_cpu_tag[slot].load(std::memory_order_acquire) != tag) {
		std::printf("      capture: slot reused\n");
		return;
	}
	const auto& cpu = g_capture_cpu[slot];
	std::printf("      cpu: args=0x%016" PRIx64 " records=0x%016" PRIx64 " stride=%u index=0x%016" PRIx64
	            " index_size=%u index_limit=%u prim_size=%u group_step=%u max_total=%u\n",
	            cpu.guest_args, cpu.records, cpu.stride, cpu.index_address, cpu.index_size,
	            cpu.index_limit, cpu.primitive_size, cpu.group_step, cpu.max_total);
	const volatile uint32_t* gpu = g_capture + static_cast<size_t>(slot) * kCaptureSlotDwords;
	if (gpu[0] != tag) {
		std::printf("      gpu: task shader had not written its record (slot tag 0x%08x)\n", gpu[0]);
		return;
	}
	std::printf("      gpu: record=[%u %u %u %u %u] (hex %08x %08x %08x %08x %08x) groups=%u launched=%u\n",
	            gpu[1], gpu[2], gpu[3], gpu[4], gpu[5], gpu[1], gpu[2], gpu[3], gpu[4], gpu[5], gpu[6],
	            gpu[7]);
}

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
	static_cast<void>(device.bindBufferMemory(g_buffer, g_memory, 0));
	void* mapped = nullptr;
	if (device.mapMemory(g_memory, 0, 64, {}, &mapped) != vk::Result::eSuccess) return;
	g_markers = static_cast<volatile uint32_t*>(mapped);
	g_markers[0] = 0;
	g_markers[1] = 0;
	g_enabled    = true;
	std::printf("GPU breadcrumbs: enabled\n");
	InitMeshCapture(physical_device, device);
}

uint64_t ReserveMeshCapture(const MeshCapture& cpu, uint32_t* tag) {
	*tag = 0;
	if (!g_enabled || g_capture == nullptr) return 0;
	uint32_t value = g_capture_next.fetch_add(1, std::memory_order_relaxed) + 1u;
	value &= 0x7fffffffu;
	if (value == 0) value = 1;
	const uint32_t slot = value % kCaptureSlots;
	g_capture_cpu_tag[slot].store(0, std::memory_order_relaxed);
	g_capture_cpu[slot] = cpu;
	g_capture_cpu_tag[slot].store(value, std::memory_order_release);
	*tag = value;
	return g_capture_address + uint64_t {slot} * kCaptureSlotDwords * 4u;
}

bool Enabled() {
	return g_enabled;
}

uint32_t Begin(vk::CommandBuffer command, const char* kind, uint64_t hash_a, uint64_t hash_b,
               uint64_t info) {
	if (!g_enabled) return 0;
	uint32_t sequence = g_next.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (sequence == 0) sequence = g_next.fetch_add(1, std::memory_order_relaxed) + 1u;
	auto& record = g_ring[sequence % kRingSize];
	record.kind   = kind;
	record.hash_a = hash_a;
	record.hash_b = hash_b;
	record.info   = info;
	record.sequence.store(sequence, std::memory_order_release);
	g_write(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_buffer, 0, sequence);
	return sequence;
}

void End(vk::CommandBuffer command, uint32_t sequence) {
	if (!g_enabled || sequence == 0) return;
	g_write(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_buffer, 4, sequence);
}

void Report() {
	static std::atomic<bool> reported {false};
	if (reported.exchange(true)) return;
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
		std::printf("  %s #%u %s 0x%016" PRIx64 " 0x%016" PRIx64 " info=0x%" PRIx64 "\n", note, sequence,
		            record.kind != nullptr ? record.kind : "?", record.hash_a, record.hash_b, record.info);
		if (record.kind != nullptr && std::strcmp(record.kind, "draw-mesh") == 0 &&
		    (record.info >> 63u) != 0) {
			PrintMeshCapture(static_cast<uint32_t>(record.info >> 32u) & 0x7fffffffu);
		}
	};
	for (uint32_t s = finished > 4 ? finished - 4 : 1; s <= finished && s != 0; s++) print(s, "done   ");
	for (uint32_t s = finished + 1; s <= finished + 12 && s <= recorded; s++) {
		print(s, s <= begun ? "RUNNING" : "queued ");
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics::GpuBreadcrumbs
