// amd_repro: compiles Kyty's dumped compute shaders (*.spv) with the Vulkan driver outside the
// emulator, to tell a driver bug from something in Kyty's process.
//
//   amd_repro.exe <folder with .spv files>     runs every CS_*.spv in every variant
//
// Each (shader, variant) pair runs in a child process, so a driver crash is reported as that
// child's exit code instead of ending the test. Results go to the console and to
// amd_repro_results.txt next to the .spv files.

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <atomic>
#include <map>
#include <thread>
#include <string>
#include <vector>

namespace {

// Variant bits.
constexpr uint32_t kFeatures = 1u << 0; // Kyty's device features
constexpr uint32_t kSubgroup = 1u << 1; // requiredSubgroupSize = 64
constexpr uint32_t kPushDesc = 1u << 2; // push-descriptor set layout
constexpr uint32_t kCache    = 1u << 3; // VkPipelineCache
constexpr uint32_t kRobust2  = 1u << 4; // VK_EXT_robustness2 features
constexpr uint32_t kAllExt   = 1u << 5; // Kyty's optional device extensions and their features
constexpr uint32_t kReserve  = 1u << 6; // Kyty's guest address-space reservation (memory.cpp)
constexpr uint32_t kReserveLow = 1u << 7; // the same, only below 4 GiB
constexpr uint32_t kCacheData = 1u << 8;  // vkGetPipelineCacheData after the pipeline
constexpr uint32_t kCacheRace  = 1u << 9;  // ... from a second thread while it is created
constexpr uint32_t kLikeKyty = kFeatures | kSubgroup | kPushDesc | kCache | kRobust2;

struct Variant {
	uint32_t    bits;
	const char* name;
};
// Every variant enables Kyty's device features: its shaders declare capabilities that need them.
constexpr Variant kVariants[] = {
    {kLikeKyty, "like-kyty"},
    {kLikeKyty | kCacheData, "like-kyty+cache-data"},
    {kLikeKyty | kCacheRace, "like-kyty+cache-data-thread"},
    {kLikeKyty | kCacheData | kCacheRace | kAllExt | kReserve, "everything"},
};

// Reserves every free block over 16 MiB in Kyty's guest ranges, as ReserveGuestRegions does.
void ReserveLikeKyty(bool below_4gib_only) {
	const std::pair<uint64_t, uint64_t> all[] = {
	    {0x0000040000ull, 0x1000000000ull},   // HOST_SYSTEM_MANAGED_MIN .. RESERVED_MAX
	    {0x1000000000ull, 0xfc00000000ull},   // HOST_USER_MIN .. HOST_USER_MAX
	    {0x080000000000ull, 0x080000000000ull + (512ull << 30)}, // extended memory
	};
	const std::pair<uint64_t, uint64_t> low[] = {{0x0000040000ull, 0x100000000ull}};
	const auto* regions = below_4gib_only ? low : all;
	const size_t count  = below_4gib_only ? 1 : 3;
	constexpr uint64_t granularity = 0x10000, minimum = 0x1000000;
	uint64_t reserved = 0;
	for (size_t r = 0; r < count; r++) {
		const auto [start, end] = regions[r];
		uint64_t next = start;
		while (next < end) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(next), &info, sizeof(info)) == 0) break;
			const auto base = reinterpret_cast<uint64_t>(info.BaseAddress);
			const auto region_end = std::min<uint64_t>(end, base + info.RegionSize);
			const auto from = (std::max(base, start) + granularity - 1) & ~(granularity - 1);
			const auto to   = region_end & ~(granularity - 1);
			if (info.State == MEM_FREE && to > from && to - from > minimum) {
				if (VirtualAlloc(reinterpret_cast<void*>(from), to - from, MEM_RESERVE, PAGE_NOACCESS))
					reserved += to - from;
			}
			next = region_end;
		}
	}
	char text[96];
	std::snprintf(text, sizeof(text), "reserved %llu GiB of address space",
	              static_cast<unsigned long long>(reserved >> 30));
	std::printf("    %s\n", text);
	std::fflush(stdout);
}

void Step(const char* text) {
	std::printf("    %s\n", text);
	std::fflush(stdout);
}

[[noreturn]] void Fail(const char* what, VkResult result) {
	std::printf("    FAILED: %s (VkResult %d)\n", what, static_cast<int>(result));
	std::fflush(stdout);
	ExitProcess(2);
}

std::vector<uint32_t> ReadSpirv(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	std::vector<uint32_t> words(bytes.size() / 4);
	std::memcpy(words.data(), bytes.data(), words.size() * 4);
	return words;
}

struct Binding {
	uint32_t         set   = 0;
	uint32_t         index = 0;
	VkDescriptorType type  = VK_DESCRIPTOR_TYPE_MAX_ENUM;
	uint32_t         count = 1;
};

struct ShaderInfo {
	bool                 compute = false;
	uint32_t             push_constant_size = 0;
	std::vector<Binding> bindings;
};

// Reads the descriptor bindings of the shader's interface from its SPIR-V.
ShaderInfo ParseSpirv(const std::vector<uint32_t>& words) {
	ShaderInfo info;
	struct TypeInfo {
		uint32_t op = 0, a = 0, b = 0, c = 0;
	};
	std::map<uint32_t, TypeInfo> types;
	std::map<uint32_t, uint32_t> constants, sets, bindings;
	std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>> variables; // id, (type, class)
	for (size_t i = 5; i < words.size();) {
		const uint32_t op = words[i] & 0xffffu, count = words[i] >> 16;
		if (count == 0 || i + count > words.size()) break;
		const uint32_t* w = &words[i];
		switch (op) {
			case 15: // OpEntryPoint
				if (w[1] == 5) info.compute = true; // GLCompute
				break;
			case 71: // OpDecorate
				if (w[2] == 34) sets[w[1]] = w[3];
				if (w[2] == 33) bindings[w[1]] = w[3];
				break;
			case 25: types[w[1]] = {25, w[7], 0, 0}; break;                   // image: sampled
			case 26: types[w[1]] = {26}; break;                               // sampler
			case 27: types[w[1]] = {27}; break;                               // sampled image
			case 28: types[w[1]] = {28, w[2], w[3]}; break;                   // array
			case 29: types[w[1]] = {29, w[2]}; break;                         // runtime array
			case 30: types[w[1]] = {30}; break;                               // struct
			case 32: types[w[1]] = {32, w[2], w[3]}; break;                   // pointer
			case 5341: types[w[1]] = {5341}; break;                           // accel. structure
			case 43: constants[w[2]] = w[3]; break;                           // OpConstant
			case 59: variables.push_back({w[2], {w[1], w[3]}}); break;        // OpVariable
			default: break;
		}
		i += count;
	}
	for (const auto& [id, type_class]: variables) {
		const auto [pointer_type, storage_class] = type_class;
		if (storage_class == 9) { // PushConstant
			info.push_constant_size = 128;
			continue;
		}
		if (!bindings.count(id)) continue;
		Binding binding;
		binding.set   = sets.count(id) ? sets[id] : 0;
		binding.index = bindings[id];
		uint32_t type = types[pointer_type].b;
		if (types[type].op == 28) {
			binding.count = constants[types[type].b];
			type          = types[type].a;
		} else if (types[type].op == 29) {
			binding.count = 16;
			type          = types[type].a;
		}
		const auto& t = types[type];
		if (storage_class == 12) binding.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		else if (storage_class == 2) binding.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		else if (t.op == 25) binding.type = t.a == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
		                                             : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		else if (t.op == 26) binding.type = VK_DESCRIPTOR_TYPE_SAMPLER;
		else if (t.op == 27) binding.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		else if (t.op == 5341) binding.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		else continue;
		// Aliased declarations (the tiler's uint/uint8/uint16 views) share one binding.
		bool seen = false;
		for (const auto& b: info.bindings) seen |= b.set == binding.set && b.index == binding.index;
		if (!seen) info.bindings.push_back(binding);
	}
	return info;
}

// Formats tried for the 3D storage image in the dispatch test.
constexpr VkFormat kImageFormats[] = {
    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16_SFLOAT,
    VK_FORMAT_R32_SFLOAT,
};
constexpr const char* kImageFormatNames[] = {"RGBA16F", "RGBA32F", "RGBA8", "R11G11B10F",
                                             "RG16F", "R16F", "R32F"};
// Image create-flag sets: Kyty's, then without 2D_ARRAY_COMPATIBLE, then none.
constexpr const char* kImageFlagNames[] = {"kyty-flags", "no-2d-array-compat", "no-flags"};

// Transfer operations on a 3D image like the ones Kyty records right after creating one
// (Image::Upload, TextureCache clears): 0 barrier only, 1 copy from a buffer, 2 clear,
// 3 clear with a sync1 barrier, 4 copy + clear + barrier back to GENERAL.
constexpr const char* kOpNames[] = {"barrier2", "copy-buffer-to-image", "clear", "clear-sync1",
                                    "copy+clear+general"};

// Specialization constants 0 and 1 (the tiler's ELEMENT_BYTES and TILE), when set.
bool     g_specialize = false;
// Set in reduce mode: the crash comes from vkCreateComputePipelines itself, no need to wait.
const bool g_fast = std::getenv("AMD_REPRO_FAST") != nullptr;
uint32_t g_spec_values[2] {};

int RunChild(const std::filesystem::path& spv, uint32_t bits, int image_format = -1,
             uint32_t image_flags = 0, int op = -1) {
	const auto words = ReadSpirv(spv);
	const auto shader = ParseSpirv(words);
	if (!shader.compute) {
		Step("not a compute shader, skipped");
		return 3;
	}

	HMODULE loader = LoadLibraryA("vulkan-1.dll");
	if (loader == nullptr) {
		Step("vulkan-1.dll not found");
		return 4;
	}
	// Kyty reserves its guest address space before it creates the Vulkan instance, which is
	// when the driver DLL loads.
	if (bits & (kReserve | kReserveLow)) ReserveLikeKyty((bits & kReserve) == 0);
	auto get_instance_proc =
	    reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(loader, "vkGetInstanceProcAddr"));
#define INSTANCE_FN(name) auto name = reinterpret_cast<PFN_##name>(get_instance_proc(instance, #name))
	VkInstance instance = VK_NULL_HANDLE;
	INSTANCE_FN(vkCreateInstance);

	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	// AMD_REPRO_NAME=Kyty gives the driver the application and engine names Kyty uses.
	const char* name_override = std::getenv("AMD_REPRO_NAME");
	app.pApplicationName = name_override != nullptr ? name_override : "amd_repro";
	app.applicationVersion = 1;
	app.pEngineName = name_override;
	app.engineVersion = 1;
	app.apiVersion       = VK_API_VERSION_1_3;
	VkInstanceCreateInfo instance_info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instance_info.pApplicationInfo = &app;
	if (auto r = vkCreateInstance(&instance_info, nullptr, &instance); r != VK_SUCCESS)
		Fail("vkCreateInstance", r);

	INSTANCE_FN(vkEnumeratePhysicalDevices);
	INSTANCE_FN(vkGetPhysicalDeviceProperties);
	INSTANCE_FN(vkGetPhysicalDeviceFeatures2);
	INSTANCE_FN(vkGetPhysicalDeviceQueueFamilyProperties);
	INSTANCE_FN(vkEnumerateDeviceExtensionProperties);
	INSTANCE_FN(vkCreateDevice);
	INSTANCE_FN(vkGetDeviceProcAddr);

	uint32_t device_count = 0;
	vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
	std::vector<VkPhysicalDevice> devices(device_count);
	vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties properties {};
	for (auto candidate: devices) {
		vkGetPhysicalDeviceProperties(candidate, &properties);
		if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			physical = candidate;
			break;
		}
	}
	if (physical == VK_NULL_HANDLE && !devices.empty()) physical = devices[0];
	if (physical == VK_NULL_HANDLE) {
		Step("no Vulkan device");
		return 4;
	}
	vkGetPhysicalDeviceProperties(physical, &properties);

	uint32_t family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
	std::vector<VkQueueFamilyProperties> families(family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
	uint32_t family = 0;
	for (uint32_t i = 0; i < family_count; i++) {
		if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			family = i;
			break;
		}
	}

	uint32_t extension_count = 0;
	vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count, nullptr);
	std::vector<VkExtensionProperties> available(extension_count);
	vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count, available.data());
	const auto has = [&](const char* name) {
		for (const auto& e: available)
			if (std::strcmp(e.extensionName, name) == 0) return true;
		return false;
	};

	// Supported features, to enable only what the device has.
	VkPhysicalDeviceRobustness2FeaturesEXT s_robust {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
	VkPhysicalDeviceVulkan13Features s13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &s_robust};
	VkPhysicalDeviceVulkan12Features s12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &s13};
	VkPhysicalDeviceVulkan11Features s11 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &s12};
	VkPhysicalDeviceFeatures2        s2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &s11};
	if (!has(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME)) s13.pNext = nullptr;
	vkGetPhysicalDeviceFeatures2(physical, &s2);

	std::vector<const char*> extensions;
	VkPhysicalDeviceRobustness2FeaturesEXT robust {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
	VkPhysicalDeviceVulkan13Features f13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	VkPhysicalDeviceVulkan12Features f12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
	VkPhysicalDeviceVulkan11Features f11 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
	VkPhysicalDeviceFeatures2        f2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
	if (bits & kFeatures) {
		// What Kyty enables (vulkanWindow.cpp), where the device supports it.
		auto& f = f2.features;
		const auto& s = s2.features;
		f.fragmentStoresAndAtomics = s.fragmentStoresAndAtomics;
		f.samplerAnisotropy = s.samplerAnisotropy;
		f.robustBufferAccess = s.robustBufferAccess;
		f.depthBounds = s.depthBounds;
		f.depthClamp = s.depthClamp;
		f.shaderStorageImageWriteWithoutFormat = s.shaderStorageImageWriteWithoutFormat;
		f.shaderStorageImageMultisample = s.shaderStorageImageMultisample;
		f.shaderImageGatherExtended = s.shaderImageGatherExtended;
		f.shaderResourceMinLod = s.shaderResourceMinLod;
		f.independentBlend = s.independentBlend;
		f.dualSrcBlend = s.dualSrcBlend;
		f.tessellationShader = s.tessellationShader;
		f.drawIndirectFirstInstance = s.drawIndirectFirstInstance;
		f.multiDrawIndirect = s.multiDrawIndirect;
		f.sampleRateShading = s.sampleRateShading;
		f.depthBiasClamp = s.depthBiasClamp;
		f.shaderClipDistance = s.shaderClipDistance;
		f.shaderCullDistance = s.shaderCullDistance;
		f.largePoints = s.largePoints;
		f.multiViewport = s.multiViewport;
		f.fillModeNonSolid = s.fillModeNonSolid;
		f.vertexPipelineStoresAndAtomics = s.vertexPipelineStoresAndAtomics;
		f.shaderInt64 = s.shaderInt64;
		f.shaderFloat64 = s.shaderFloat64;
		f11.storageBuffer16BitAccess = s11.storageBuffer16BitAccess;
		f11.shaderDrawParameters = s11.shaderDrawParameters;
		f12.samplerMirrorClampToEdge = s12.samplerMirrorClampToEdge;
		f12.storageBuffer8BitAccess = s12.storageBuffer8BitAccess;
		f12.shaderBufferInt64Atomics = s12.shaderBufferInt64Atomics;
		f12.shaderSharedInt64Atomics = s12.shaderSharedInt64Atomics;
		f12.samplerFilterMinmax = s12.samplerFilterMinmax;
		f12.shaderSampledImageArrayNonUniformIndexing = s12.shaderSampledImageArrayNonUniformIndexing;
		f12.shaderStorageBufferArrayNonUniformIndexing = s12.shaderStorageBufferArrayNonUniformIndexing;
		f12.shaderStorageImageArrayNonUniformIndexing = s12.shaderStorageImageArrayNonUniformIndexing;
		f12.descriptorBindingSampledImageUpdateAfterBind = s12.descriptorBindingSampledImageUpdateAfterBind;
		f12.descriptorBindingUpdateUnusedWhilePending = s12.descriptorBindingUpdateUnusedWhilePending;
		f12.descriptorBindingPartiallyBound = s12.descriptorBindingPartiallyBound;
		f12.timelineSemaphore = s12.timelineSemaphore;
		f12.bufferDeviceAddress = s12.bufferDeviceAddress;
		f12.shaderOutputViewportIndex = s12.shaderOutputViewportIndex;
		f12.shaderOutputLayer = s12.shaderOutputLayer;
		f13.synchronization2 = s13.synchronization2;
		f13.dynamicRendering = s13.dynamicRendering;
		f13.robustImageAccess = s13.robustImageAccess;
	}
	if (bits & kSubgroup) f13.subgroupSizeControl = s13.subgroupSizeControl;
	if ((bits & kRobust2) && has(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME)) {
		extensions.push_back(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
		robust.robustBufferAccess2 = s_robust.robustBufferAccess2;
		robust.robustImageAccess2  = s_robust.robustImageAccess2;
		robust.nullDescriptor      = s_robust.nullDescriptor;
		f13.pNext = &robust;
	}
	// Kyty's other extensions (vulkanWindow.cpp), with the features it enables for them.
	VkPhysicalDeviceMeshShaderFeaturesEXT mesh {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
	VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR};
	VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT atomic64 {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT};
	VkPhysicalDeviceMaintenance8FeaturesKHR maintenance8 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
	VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
	VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentric {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
	VkPhysicalDeviceShaderClockFeaturesKHR clock {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
	VkPhysicalDeviceDepthClipEnableFeaturesEXT clip_enable {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT};
	VkPhysicalDeviceDepthClipControlFeaturesEXT clip_control {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
	VkPhysicalDeviceImageViewMinLodFeaturesEXT min_lod {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT};
	VkPhysicalDeviceExtendedDynamicStateFeaturesEXT dynamic_state {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT};
	if (bits & kAllExt) {
		struct Extension {
			const char* name;
			void*       features;
		};
		const Extension list[] = {
		    {VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, nullptr},
		    {VK_EXT_DEPTH_RANGE_UNRESTRICTED_EXTENSION_NAME, nullptr},
		    {"VK_KHR_maintenance1", nullptr},
		    {VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, &dynamic_state},
		    {VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME, &workgroup},
		    {VK_EXT_MESH_SHADER_EXTENSION_NAME, &mesh},
		    {VK_KHR_MAINTENANCE_8_EXTENSION_NAME, &maintenance8},
		    {VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clock},
		    {VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, &atomic64},
		    {VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &barycentric},
		    {VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, &derivatives},
		    {VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME, &clip_enable},
		    {VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME, &clip_control},
		    {VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME, &min_lod},
		};
		// Query what the device supports through the same structs, then enable it.
		VkBaseOutStructure* chain = nullptr;
		for (const auto& e: list) {
			if (!has(e.name)) continue;
			extensions.push_back(e.name);
			if (e.features != nullptr) {
				auto* base  = static_cast<VkBaseOutStructure*>(e.features);
				base->pNext = chain;
				chain       = base;
			}
		}
		if (chain != nullptr) {
			VkPhysicalDeviceFeatures2 query {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, chain};
			vkGetPhysicalDeviceFeatures2(physical, &query);
			// As Kyty: no mesh multiview/VRS, no sparse int64 atomics.
			mesh.multiviewMeshShader = VK_FALSE;
			mesh.primitiveFragmentShadingRateMeshShader = VK_FALSE;
			mesh.meshShaderQueries = VK_FALSE;
			atomic64.sparseImageInt64Atomics = VK_FALSE;
			auto* tail = static_cast<VkBaseOutStructure*>(f2.pNext);
			while (tail->pNext != nullptr) tail = tail->pNext;
			tail->pNext = chain;
		}
	}
	const bool push_desc = (bits & kPushDesc) && has(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
	if (push_desc) extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

	const float priority = 1.0f;
	VkDeviceQueueCreateInfo queue_info {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queue_info.queueFamilyIndex = family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;
	VkDeviceCreateInfo device_info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue_info;
	device_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
	device_info.ppEnabledExtensionNames = extensions.data();
	VkDevice device = VK_NULL_HANDLE;
	Step("creating device");
	if (auto r = vkCreateDevice(physical, &device_info, nullptr, &device); r != VK_SUCCESS)
		Fail("vkCreateDevice", r);

#define DEVICE_FN(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(device, #name))
	DEVICE_FN(vkCreateShaderModule);
	DEVICE_FN(vkCreateDescriptorSetLayout);
	DEVICE_FN(vkCreatePipelineLayout);
	DEVICE_FN(vkCreatePipelineCache);
	DEVICE_FN(vkCreateComputePipelines);
	DEVICE_FN(vkDestroyPipeline);
	DEVICE_FN(vkDeviceWaitIdle);

	VkShaderModuleCreateInfo module_info {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	module_info.codeSize = words.size() * 4;
	module_info.pCode    = words.data();
	VkShaderModule module = VK_NULL_HANDLE;
	Step("creating shader module");
	if (auto r = vkCreateShaderModule(device, &module_info, nullptr, &module); r != VK_SUCCESS)
		Fail("vkCreateShaderModule", r);
	if (!g_fast) Sleep(1500); // a driver thread that crashes on the module does so here

	uint32_t max_set = 0;
	for (const auto& b: shader.bindings) max_set = b.set > max_set ? b.set : max_set;
	std::vector<VkDescriptorSetLayout> set_layouts(shader.bindings.empty() ? 0 : max_set + 1);
	for (uint32_t set = 0; set < set_layouts.size(); set++) {
		std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
		for (const auto& b: shader.bindings) {
			if (b.set != set) continue;
			layout_bindings.push_back({b.index, b.type, b.count, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
		}
		VkDescriptorSetLayoutCreateInfo layout_info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		if (push_desc && set == 0) layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
		layout_info.bindingCount = static_cast<uint32_t>(layout_bindings.size());
		layout_info.pBindings    = layout_bindings.data();
		if (auto r = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &set_layouts[set]);
		    r != VK_SUCCESS)
			Fail("vkCreateDescriptorSetLayout", r);
	}
	// Kyty always declares its 128-byte push-constant block (NativePushConstantSize).
	VkPushConstantRange push_range {VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
	VkPipelineLayoutCreateInfo pipeline_layout_info {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	pipeline_layout_info.setLayoutCount         = static_cast<uint32_t>(set_layouts.size());
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	if (auto r = vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout);
	    r != VK_SUCCESS)
		Fail("vkCreatePipelineLayout", r);

	VkPipelineCache cache = VK_NULL_HANDLE;
	if (bits & kCache) {
		VkPipelineCacheCreateInfo cache_info {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
		vkCreatePipelineCache(device, &cache_info, nullptr, &cache);
	}

	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup {
	    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
	subgroup.requiredSubgroupSize = 64;
	VkComputePipelineCreateInfo pipeline_info {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
	pipeline_info.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	if ((bits & kSubgroup) && s13.subgroupSizeControl) pipeline_info.stage.pNext = &subgroup;
	const VkSpecializationMapEntry spec_entries[] {{0, 0, 4}, {1, 4, 4}};
	const VkSpecializationInfo     spec_info {2, spec_entries, sizeof(g_spec_values), g_spec_values};
	if (g_specialize) pipeline_info.stage.pSpecializationInfo = &spec_info;
	pipeline_info.layout = pipeline_layout;
	VkPipeline pipeline = VK_NULL_HANDLE;
	DEVICE_FN(vkGetPipelineCacheData);
	const auto read_cache = [&] {
		size_t size = 0;
		if (vkGetPipelineCacheData(device, cache, &size, nullptr) != VK_SUCCESS || size == 0) return;
		std::vector<uint8_t> data(size);
		vkGetPipelineCacheData(device, cache, &size, data.data());
	};
	// Kyty saves the cache on its store thread while the GPU thread goes on creating pipelines.
	std::atomic<bool> racing {true};
	std::thread       racer;
	if ((bits & kCacheRace) && cache != VK_NULL_HANDLE) {
		Step("reading the pipeline cache from a second thread");
		racer = std::thread([&] {
			while (racing.load()) read_cache();
		});
	}
	Step("creating compute pipeline");
	const auto pipeline_result =
	    vkCreateComputePipelines(device, cache, 1, &pipeline_info, nullptr, &pipeline);
	if ((bits & kCacheRace) && racer.joinable()) {
		Sleep(500);
		racing.store(false);
		racer.join();
	}
	if (pipeline_result != VK_SUCCESS) Fail("vkCreateComputePipelines", pipeline_result);
	if (g_fast) return 0;
	if ((bits & kCacheData) && cache != VK_NULL_HANDLE) {
		Step("reading the pipeline cache");
		read_cache();
	}
	Sleep(1500);
	vkDeviceWaitIdle(device);
	Step("pipeline created OK");
	if (image_format < 0) {
		vkDestroyPipeline(device, pipeline, nullptr);
		return 0;
	}

	// Dispatch test: a real 3D storage image made the way Kyty makes it (image.cpp), bound with
	// push descriptors, written by the shader on the GPU.
	DEVICE_FN(vkCreateImage);
	DEVICE_FN(vkGetImageMemoryRequirements);
	DEVICE_FN(vkAllocateMemory);
	DEVICE_FN(vkBindImageMemory);
	DEVICE_FN(vkCreateImageView);
	DEVICE_FN(vkCreateBuffer);
	DEVICE_FN(vkGetBufferMemoryRequirements);
	DEVICE_FN(vkBindBufferMemory);
	DEVICE_FN(vkCreateCommandPool);
	DEVICE_FN(vkAllocateCommandBuffers);
	DEVICE_FN(vkBeginCommandBuffer);
	DEVICE_FN(vkEndCommandBuffer);
	DEVICE_FN(vkCmdPipelineBarrier);
	DEVICE_FN(vkCmdBindPipeline);
	DEVICE_FN(vkCmdPushConstants);
	DEVICE_FN(vkCmdDispatch);
	DEVICE_FN(vkGetDeviceQueue);
	DEVICE_FN(vkQueueSubmit);
	DEVICE_FN(vkQueueWaitIdle);
	DEVICE_FN(vkCreateDescriptorPool);
	DEVICE_FN(vkAllocateDescriptorSets);
	DEVICE_FN(vkUpdateDescriptorSets);
	DEVICE_FN(vkCmdBindDescriptorSets);
	auto vkCmdPushDescriptorSetKHR =
	    reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR"));
	INSTANCE_FN(vkGetPhysicalDeviceMemoryProperties);
	VkPhysicalDeviceMemoryProperties memory_properties {};
	vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
	const auto memory_type = [&](uint32_t type_bits) {
		for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
			if ((type_bits & (1u << i)) &&
			    (memory_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
				return i;
		}
		for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
			if (type_bits & (1u << i)) return i;
		return 0u;
	};

	const VkFormat format = kImageFormats[image_format];
	VkImageCreateInfo image_info {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	if (image_flags == 0) {
		image_info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT |
		                   VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
	} else if (image_flags == 1) {
		image_info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
	}
	image_info.imageType   = VK_IMAGE_TYPE_3D;
	image_info.format      = format;
	image_info.extent      = op >= 0 ? VkExtent3D {256, 256, 4} : VkExtent3D {64, 64, 32};
	image_info.mipLevels   = 1;
	image_info.arrayLayers = 1;
	image_info.samples     = VK_SAMPLE_COUNT_1_BIT;
	image_info.tiling      = VK_IMAGE_TILING_OPTIMAL;
	// Usage as Kyty picks it (ImageUsageFlags): from the format's optimal-tiling features.
	INSTANCE_FN(vkGetPhysicalDeviceFormatProperties);
	VkFormatProperties format_properties {};
	vkGetPhysicalDeviceFormatProperties(physical, format, &format_properties);
	const auto features = format_properties.optimalTilingFeatures;
	if (!(features & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
		Step("format has no storage-image support, skipped");
		return 3;
	}
	image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	                   VK_IMAGE_USAGE_STORAGE_BIT;
	if (features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) image_info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
	if (features & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)
		image_info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImage image = VK_NULL_HANDLE;
	Step("creating 3D image");
	if (auto r = vkCreateImage(device, &image_info, nullptr, &image); r != VK_SUCCESS)
		Fail("vkCreateImage", r);
	VkMemoryRequirements image_requirements {};
	vkGetImageMemoryRequirements(device, image, &image_requirements);
	VkMemoryAllocateInfo allocate {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	allocate.allocationSize  = image_requirements.size;
	allocate.memoryTypeIndex = memory_type(image_requirements.memoryTypeBits);
	VkDeviceMemory image_memory = VK_NULL_HANDLE;
	if (auto r = vkAllocateMemory(device, &allocate, nullptr, &image_memory); r != VK_SUCCESS)
		Fail("vkAllocateMemory(image)", r);
	vkBindImageMemory(device, image, image_memory, 0);

	VkImageViewUsageCreateInfo view_usage {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
	view_usage.usage = VK_IMAGE_USAGE_STORAGE_BIT;
	VkImageViewCreateInfo view_info {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	if (image_flags != 2) view_info.pNext = &view_usage;
	view_info.image            = image;
	view_info.viewType         = VK_IMAGE_VIEW_TYPE_3D;
	view_info.format           = format;
	view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	VkImageView view = VK_NULL_HANDLE;
	Step("creating 3D image view");
	if (auto r = vkCreateImageView(device, &view_info, nullptr, &view); r != VK_SUCCESS)
		Fail("vkCreateImageView", r);

	VkBufferCreateInfo buffer_info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer_info.size  = 65536;
	buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkBuffer buffer = VK_NULL_HANDLE;
	if (auto r = vkCreateBuffer(device, &buffer_info, nullptr, &buffer); r != VK_SUCCESS)
		Fail("vkCreateBuffer", r);
	VkMemoryRequirements buffer_requirements {};
	vkGetBufferMemoryRequirements(device, buffer, &buffer_requirements);
	allocate.allocationSize  = buffer_requirements.size;
	allocate.memoryTypeIndex = memory_type(buffer_requirements.memoryTypeBits);
	VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
	if (auto r = vkAllocateMemory(device, &allocate, nullptr, &buffer_memory); r != VK_SUCCESS)
		Fail("vkAllocateMemory(buffer)", r);
	vkBindBufferMemory(device, buffer, buffer_memory, 0);

	VkCommandPoolCreateInfo pool_info {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	pool_info.queueFamilyIndex = family;
	VkCommandPool pool = VK_NULL_HANDLE;
	vkCreateCommandPool(device, &pool_info, nullptr, &pool);
	VkCommandBufferAllocateInfo command_info {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	command_info.commandPool        = pool;
	command_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	command_info.commandBufferCount = 1;
	VkCommandBuffer command = VK_NULL_HANDLE;
	vkAllocateCommandBuffers(device, &command_info, &command);
	VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	vkBeginCommandBuffer(command, &begin);
	if (op >= 0) {
		DEVICE_FN(vkCmdPipelineBarrier2);
		DEVICE_FN(vkCmdCopyBufferToImage);
		DEVICE_FN(vkCmdClearColorImage);
		const VkImageSubresourceRange range {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		const auto barrier2 = [&](VkImageLayout from, VkImageLayout to) {
			VkImageMemoryBarrier2 b {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
			b.srcStageMask        = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			b.srcAccessMask       = VK_ACCESS_2_MEMORY_WRITE_BIT;
			b.dstStageMask        = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			b.dstAccessMask       = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
			b.oldLayout           = from;
			b.newLayout           = to;
			b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			b.image               = image;
			b.subresourceRange    = range;
			VkDependencyInfo dependency {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
			dependency.dependencyFlags         = VK_DEPENDENCY_BY_REGION_BIT;
			dependency.imageMemoryBarrierCount = 1;
			dependency.pImageMemoryBarriers    = &b;
			vkCmdPipelineBarrier2(command, &dependency);
		};
		if (op == 3) {
			VkImageMemoryBarrier b {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
			b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
			b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
			b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			b.image               = image;
			b.subresourceRange    = range;
			vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
		} else {
			barrier2(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		}
		if (op == 1 || op == 4) {
			// The buffer holds 64 KiB; copy the rows that fit, a full-width 3D region.
			const uint32_t texel = format == VK_FORMAT_R16G16B16A16_SFLOAT ? 8u : 4u;
			const uint32_t rows  = 65536u / (256u * texel);
			VkBufferImageCopy copy {};
			copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
			copy.imageExtent      = {256, rows, 1};
			vkCmdCopyBufferToImage(command, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
			                       &copy);
		}
		if (op == 2 || op == 3 || op == 4) {
			VkClearColorValue color {};
			color.float32[0] = 1.0f;
			vkCmdClearColorImage(command, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
		}
		if (op == 4) barrier2(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
		vkEndCommandBuffer(command);
		VkQueue queue = VK_NULL_HANDLE;
		vkGetDeviceQueue(device, family, 0, &queue);
		VkSubmitInfo submit {VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers    = &command;
		Step("submitting transfer operations");
		if (auto r = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE); r != VK_SUCCESS)
			Fail("vkQueueSubmit", r);
		if (auto r = vkQueueWaitIdle(queue); r != VK_SUCCESS) Fail("vkQueueWaitIdle", r);
		Sleep(1500);
		Step("transfer operations finished OK");
		return 0;
	}
	VkImageMemoryBarrier barrier {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
	barrier.srcAccessMask       = 0;
	barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image;
	barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
	vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

	std::vector<VkWriteDescriptorSet> writes;
	std::vector<VkDescriptorBufferInfo> buffer_infos(shader.bindings.size());
	std::vector<VkDescriptorImageInfo>  image_infos(shader.bindings.size());
	for (size_t i = 0; i < shader.bindings.size(); i++) {
		const auto& b = shader.bindings[i];
		if (b.set != 0) continue;
		VkWriteDescriptorSet write {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		write.dstBinding      = b.index;
		write.descriptorCount = 1;
		write.descriptorType  = b.type;
		if (b.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || b.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
			buffer_infos[i]     = {buffer, 0, VK_WHOLE_SIZE};
			write.pBufferInfo   = &buffer_infos[i];
		} else if (b.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
			image_infos[i]      = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
			write.pImageInfo    = &image_infos[i];
		} else {
			continue;
		}
		writes.push_back(write);
	}
	if (push_desc && vkCmdPushDescriptorSetKHR != nullptr) {
		vkCmdPushDescriptorSetKHR(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0,
		                          static_cast<uint32_t>(writes.size()), writes.data());
	} else {
		VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8},
		                                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8}};
		VkDescriptorPoolCreateInfo descriptor_pool_info {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		descriptor_pool_info.maxSets       = 1;
		descriptor_pool_info.poolSizeCount = 2;
		descriptor_pool_info.pPoolSizes    = sizes;
		VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
		vkCreateDescriptorPool(device, &descriptor_pool_info, nullptr, &descriptor_pool);
		VkDescriptorSetAllocateInfo set_info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		set_info.descriptorPool     = descriptor_pool;
		set_info.descriptorSetCount = 1;
		set_info.pSetLayouts        = &set_layouts[0];
		VkDescriptorSet set = VK_NULL_HANDLE;
		vkAllocateDescriptorSets(device, &set_info, &set);
		for (auto& w: writes) w.dstSet = set;
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
		vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set,
		                        0, nullptr);
	}
	const uint32_t push_data[32] = {};
	vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data),
	                   push_data);
	vkCmdDispatch(command, 8, 8, 32);
	vkEndCommandBuffer(command);
	VkQueue queue = VK_NULL_HANDLE;
	vkGetDeviceQueue(device, family, 0, &queue);
	VkSubmitInfo submit {VK_STRUCTURE_TYPE_SUBMIT_INFO};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	Step("submitting dispatch");
	if (auto r = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE); r != VK_SUCCESS)
		Fail("vkQueueSubmit", r);
	if (auto r = vkQueueWaitIdle(queue); r != VK_SUCCESS) Fail("vkQueueWaitIdle", r);
	Sleep(1000);
	Step("dispatch finished OK");
	return 0;
}


// ---- Automatic reduction (--reduce <shader.spv>) ----
// Replaces value-producing instructions with OpCopyObject of an OpConstantNull, in ever smaller
// chunks, keeping every replacement after which the driver still crashes. Each variant stays
// valid SPIR-V: only results of scalar, vector, matrix, array and struct types are replaced.

using Instruction = std::vector<uint32_t>;

struct Module {
	std::vector<uint32_t>    header;
	std::vector<Instruction> instructions;
};

Module SplitSpirv(const std::vector<uint32_t>& words) {
	Module module;
	module.header.assign(words.begin(), words.begin() + 5);
	for (size_t i = 5; i < words.size();) {
		const uint32_t count = words[i] >> 16;
		if (count == 0 || i + count > words.size()) break;
		module.instructions.emplace_back(words.begin() + i, words.begin() + i + count);
		i += count;
	}
	return module;
}

bool IsValueOpcode(uint32_t op) {
	return op == 12 || op == 57 || op == 61 || (op >= 79 && op <= 84) || (op >= 87 && op <= 107) ||
	       (op >= 109 && op <= 152) || (op >= 154 && op <= 205) || (op >= 333 && op <= 366);
}

struct Reducer {
	Module                    module;
	std::vector<size_t>       candidates; // instruction indices that may be replaced
	std::map<uint32_t, bool>  nullable;   // type id -> OpConstantNull allowed
	size_t                    first_function = 0;
	std::filesystem::path     work;
	std::string               self;
	uint32_t                  tests = 0;

	void Analyze() {
		std::map<uint32_t, Instruction> types;
		bool in_function = false;
		for (size_t i = 0; i < module.instructions.size(); i++) {
			const auto& in = module.instructions[i];
			const uint32_t op = in[0] & 0xffffu;
			if (op >= 19 && op <= 39 && in.size() >= 2) types[in[1]] = in;
			if (op == 54) {
				if (first_function == 0) first_function = i;
				in_function = true;
			}
			if (op == 56) in_function = false;
			if (in_function && op != 54 && in.size() >= 3 && IsValueOpcode(op) && IsNullable(types, in[1]))
				candidates.push_back(i);
		}
	}

	bool IsNullable(std::map<uint32_t, Instruction>& types, uint32_t type) {
		if (auto it = nullable.find(type); it != nullable.end()) return it->second;
		bool result = false;
		if (auto it = types.find(type); it != types.end()) {
			const auto& t  = it->second;
			const auto  op = t[0] & 0xffffu;
			if (op == 20 || op == 21 || op == 22) result = true;
			else if (op == 23 || op == 24 || op == 28) result = IsNullable(types, t[2]);
			else if (op == 30) {
				result = true;
				for (size_t m = 2; m < t.size(); m++) result = result && IsNullable(types, t[m]);
			}
		}
		nullable[type] = result;
		return result;
	}

	std::vector<uint32_t> Build(const std::vector<bool>& replaced) const {
		std::map<uint32_t, uint32_t> nulls;
		uint32_t bound = module.header[3];
		std::vector<Instruction> body = module.instructions;
		for (size_t c = 0; c < candidates.size(); c++) {
			if (!replaced[c]) continue;
			auto& in = body[candidates[c]];
			const uint32_t type = in[1], result = in[2];
			auto [it, inserted] = nulls.try_emplace(type, bound);
			if (inserted) bound++;
			in = {(4u << 16) | 83u, type, result, it->second};
		}
		std::vector<uint32_t> words = module.header;
		words[3] = bound;
		for (size_t i = 0; i < body.size(); i++) {
			if (i == first_function)
				for (const auto& [type, id]: nulls) words.insert(words.end(), {(3u << 16) | 46u, type, id});
			words.insert(words.end(), body[i].begin(), body[i].end());
		}
		return words;
	}

	bool Crashes(const std::vector<bool>& replaced, const char* save_as = nullptr) {
		const auto words = Build(replaced);
		const auto path  = work / (save_as != nullptr ? save_as : "candidate.spv");
		{
			std::ofstream out(path, std::ios::binary);
			out.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
		}
		std::string command = "\"" + self + "\" --child \"" + path.string() + "\" " +
		                      std::to_string(kFeatures | kPushDesc) + " -1 0 -1";
		STARTUPINFOA startup {sizeof(startup)};
		startup.dwFlags     = STARTF_USESTDHANDLES;
		PROCESS_INFORMATION process {};
		if (!CreateProcessA(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
		                    nullptr, nullptr, &startup, &process))
			return false;
		DWORD code = 0;
		if (WaitForSingleObject(process.hProcess, 60000) == WAIT_TIMEOUT) {
			TerminateProcess(process.hProcess, STILL_ACTIVE);
			code = STILL_ACTIVE;
		} else {
			GetExitCodeProcess(process.hProcess, &code);
		}
		CloseHandle(process.hProcess);
		CloseHandle(process.hThread);
		tests++;
		return code == 0xC0000005;
	}
};

int Reduce(const std::filesystem::path& spv) {
	SetEnvironmentVariableA("AMD_REPRO_FAST", "1");
	Reducer r;
	r.module = SplitSpirv(ReadSpirv(spv));
	r.work   = spv.parent_path();
	char self[MAX_PATH];
	GetModuleFileNameA(nullptr, self, MAX_PATH);
	r.self = self;
	r.Analyze();
	std::ofstream log(r.work / "amd_repro_reduce_log.txt");
	const auto report = [&](const std::string& line) {
		std::printf("%s\n", line.c_str());
		std::fflush(stdout);
		log << line << "\n";
		log.flush();
	};
	const size_t n = r.candidates.size();
	report("reduce: " + spv.filename().string() + ", " + std::to_string(n) +
	       " replaceable instructions of " + std::to_string(r.module.instructions.size()));
	std::vector<bool> replaced(n, false);
	if (!r.Crashes(replaced)) {
		report("the original does not crash; nothing to reduce");
		return 1;
	}
	std::vector<bool> all(n, true);
	const bool all_crash = r.Crashes(all, "reduced_all_values.spv");
	report(std::string("with every value replaced: ") + (all_crash ? "still crashes" : "no crash"));
	if (all_crash) {
		replaced = all;
	} else {
		for (size_t chunk = (n + 1) / 2; chunk >= 1; chunk = chunk == 1 ? 0 : (chunk + 1) / 2) {
			std::vector<size_t> remaining;
			for (size_t c = 0; c < n; c++)
				if (!replaced[c]) remaining.push_back(c);
			if (chunk > remaining.size()) chunk = remaining.size();
			for (size_t start = 0; start < remaining.size(); start += chunk) {
				auto trial = replaced;
				for (size_t k = start; k < std::min(start + chunk, remaining.size()); k++)
					trial[remaining[k]] = true;
				if (r.Crashes(trial)) replaced = trial;
			}
			size_t left = 0;
			for (bool b: replaced) left += b ? 0 : 1;
			report("chunk " + std::to_string(chunk) + ": " + std::to_string(left) +
			       " instructions still needed for the crash (" + std::to_string(r.tests) + " tests)");
			if (r.tests > 1500) {
				report("test budget reached");
				break;
			}
		}
	}
	r.Crashes(replaced, "reduced.spv");
	std::string kept;
	for (size_t c = 0; c < n; c++)
		if (!replaced[c]) kept += std::to_string(r.candidates[c]) + " ";
	report("kept instruction indices: " + kept);
	report("saved reduced.spv; done");
	return 0;
}

std::string Describe(DWORD code) {
	char text[64];
	switch (code) {
		case 0: return "OK";
		case 2: return "Vulkan error (see output)";
		case 3: return "skipped";
		case 4: return "no Vulkan";
		case 0xC0000005: return "CRASH (access violation)";
		case STILL_ACTIVE: return "HUNG (killed after 60 s)";
		default: std::snprintf(text, sizeof(text), "exit code 0x%08lX", code); return text;
	}
}

} // namespace

int main(int argc, char** argv) {
	if (argc >= 4 && std::strcmp(argv[1], "--child") == 0) {
		const int image_format = argc >= 6 ? std::atoi(argv[4]) : -1;
		const auto image_flags = argc >= 6 ? static_cast<uint32_t>(std::atoi(argv[5])) : 0u;
		const int op = argc >= 7 ? std::atoi(argv[6]) : -1;
		if (argc >= 9) {
			g_specialize     = true;
			g_spec_values[0] = static_cast<uint32_t>(std::atoi(argv[7]));
			g_spec_values[1] = static_cast<uint32_t>(std::atoi(argv[8]));
		}
		return RunChild(argv[2], static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)),
		                image_format, image_flags, op);
	}
	if (argc >= 3 && std::strcmp(argv[1], "--reduce") == 0) return Reduce(argv[2]);
	// amd_repro.exe --dispatch <shader.spv>: runs the shader on real 3D images.
	const bool ops      = argc >= 3 && std::strcmp(argv[1], "--ops") == 0;
	const bool tiler    = argc >= 3 && std::strcmp(argv[1], "--tiler") == 0;
	const bool dispatch = ops || (argc >= 3 && std::strcmp(argv[1], "--dispatch") == 0);
	const std::filesystem::path folder =
	    tiler ? std::filesystem::path(argv[2])
	          : dispatch ? std::filesystem::path(argv[2]).parent_path() : (argc >= 2 ? argv[1] : ".");
	std::vector<std::filesystem::path> shaders;
	if (tiler) {
		for (const auto& entry: std::filesystem::recursive_directory_iterator(folder))
			if (entry.path().extension() == ".spv") shaders.push_back(entry.path());
		std::sort(shaders.begin(), shaders.end());
	} else if (dispatch) {
		shaders.push_back(argv[2]);
	} else {
		for (const auto& entry: std::filesystem::directory_iterator(folder)) {
			const auto name = entry.path().filename().string();
			if (entry.path().extension() == ".spv" && name.rfind("CS_", 0) == 0)
				shaders.push_back(entry.path());
		}
	}
	if (shaders.empty()) {
		std::printf("No CS_*.spv files in %s\n", folder.string().c_str());
		return 1;
	}
	char self[MAX_PATH];
	GetModuleFileNameA(nullptr, self, MAX_PATH);

	std::ofstream results(folder / (tiler ? "amd_repro_tiler_results.txt"
	                                : ops ? "amd_repro_ops_results.txt"
	                                    : dispatch ? "amd_repro_dispatch_results.txt"
	                                               : "amd_repro_results.txt"));
	const auto report = [&](const std::string& line) {
		std::printf("%s\n", line.c_str());
		std::fflush(stdout);
		results << line << "\n";
		results.flush();
	};
	report("amd_repro: " + std::to_string(shaders.size()) + " compute shaders" +
	       (dispatch ? ", dispatch on real 3D images" : ""));
	struct Job {
		std::filesystem::path shader;
		uint32_t              bits;
		int                   format;
		uint32_t              flags;
		std::string           label;
		int                   op = -1;
		uint32_t              element_bytes = 0;
		uint32_t              tile = 0;
	};
	std::vector<Job> jobs;
	for (const auto& shader: shaders) {
		if (tiler) {
			// TileManager::GetPipeline: ELEMENT_BYTES = 1 << element_index, TILE = direction.
			for (const uint32_t element_bytes: {1u, 2u, 4u, 8u, 16u})
				for (const uint32_t tile: {0u, 1u})
					jobs.push_back({shader, kFeatures | kPushDesc, -1, 0,
					                "ELEMENT_BYTES=" + std::to_string(element_bytes) +
					                    " TILE=" + std::to_string(tile),
					                -1, element_bytes, tile});
		} else if (ops) {
			// Kyty's 3D images in Demon's Souls: 256x256x4, RGBA16F (format 97) and R32F (100).
			for (const int format: {0, 6})
				for (int op = 0; op < static_cast<int>(std::size(kOpNames)); op++)
					jobs.push_back({shader, kLikeKyty, format, 0,
					                std::string(kImageFormatNames[format]) + " " + kOpNames[op], op});
		} else if (dispatch) {
			for (int format = 0; format < static_cast<int>(std::size(kImageFormats)); format++)
				for (uint32_t flags = 0; flags < 3; flags++)
					jobs.push_back({shader, kLikeKyty, format, flags,
					                std::string(kImageFormatNames[format]) + " " + kImageFlagNames[flags]});
			jobs.push_back({shader, kLikeKyty & ~kPushDesc, 0, 0, "RGBA16F kyty-flags descriptor-set"});
		} else {
			for (const auto& variant: kVariants) jobs.push_back({shader, variant.bits, -1, 0, variant.name});
		}
	}
	for (const auto& job: jobs) {
		const auto& shader = job.shader;
		{
			std::printf("%s [%s]\n", shader.filename().string().c_str(), job.label.c_str());
			std::fflush(stdout);
			std::string command = std::string("\"") + self + "\" --child \"" + shader.string() + "\" " +
			                      std::to_string(job.bits) + " " + std::to_string(job.format) + " " +
			                      std::to_string(job.flags) + " " + std::to_string(job.op);
			if (job.element_bytes != 0)
				command += " " + std::to_string(job.element_bytes) + " " + std::to_string(job.tile);
			STARTUPINFOA startup {sizeof(startup)};
			PROCESS_INFORMATION process {};
			if (!CreateProcessA(nullptr, command.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr,
			                    &startup, &process)) {
				report("  could not start child process");
				continue;
			}
			DWORD code = STILL_ACTIVE;
			if (WaitForSingleObject(process.hProcess, 60000) == WAIT_TIMEOUT) {
				TerminateProcess(process.hProcess, STILL_ACTIVE);
			} else {
				GetExitCodeProcess(process.hProcess, &code);
			}
			CloseHandle(process.hProcess);
			CloseHandle(process.hThread);
			char line[512];
			const auto shown = tiler ? std::filesystem::relative(shader, folder).string()
			                         : shader.filename().string();
			std::snprintf(line, sizeof(line), "%-48s %-34s %s", shown.c_str(), job.label.c_str(),
			              Describe(code).c_str());
			report(line);
		}
	}
	report("done");
	return 0;
}
