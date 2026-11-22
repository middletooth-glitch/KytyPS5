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

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
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
constexpr uint32_t kLikeKyty = kFeatures | kSubgroup | kPushDesc | kCache | kRobust2;

struct Variant {
	uint32_t    bits;
	const char* name;
};
// Every variant enables Kyty's device features: its shaders declare capabilities that need them.
constexpr Variant kVariants[] = {
    {kLikeKyty, "like-kyty"},
    {kLikeKyty | kAllExt, "like-kyty+extensions"},
    {kLikeKyty | kReserve, "like-kyty+address-reserve"},
    {kLikeKyty | kReserveLow, "like-kyty+reserve-below-4GiB"},
    {kLikeKyty | kAllExt | kReserve, "like-kyty+ext+reserve"},
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
		info.bindings.push_back(binding);
	}
	return info;
}

int RunChild(const std::filesystem::path& spv, uint32_t bits) {
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
	app.pApplicationName = "amd_repro";
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
	Sleep(1500); // a driver thread that crashes on the module does so here

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
	pipeline_info.layout = pipeline_layout;
	VkPipeline pipeline = VK_NULL_HANDLE;
	Step("creating compute pipeline");
	if (auto r = vkCreateComputePipelines(device, cache, 1, &pipeline_info, nullptr, &pipeline);
	    r != VK_SUCCESS)
		Fail("vkCreateComputePipelines", r);
	Sleep(1500);
	vkDeviceWaitIdle(device);
	vkDestroyPipeline(device, pipeline, nullptr);
	Step("pipeline created OK");
	return 0;
}

std::string Describe(DWORD code) {
	char text[64];
	switch (code) {
		case 0: return "OK";
		case 2: return "Vulkan error (see output)";
		case 3: return "skipped (not compute)";
		case 4: return "no Vulkan";
		case 0xC0000005: return "CRASH (access violation)";
		case STILL_ACTIVE: return "HUNG (killed after 60 s)";
		default: std::snprintf(text, sizeof(text), "exit code 0x%08lX", code); return text;
	}
}

} // namespace

int main(int argc, char** argv) {
	if (argc >= 4 && std::strcmp(argv[1], "--child") == 0) {
		return RunChild(argv[2], static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)));
	}
	const std::filesystem::path folder = argc >= 2 ? argv[1] : ".";
	std::vector<std::filesystem::path> shaders;
	for (const auto& entry: std::filesystem::directory_iterator(folder)) {
		const auto name = entry.path().filename().string();
		if (entry.path().extension() == ".spv" && name.rfind("CS_", 0) == 0) shaders.push_back(entry.path());
	}
	if (shaders.empty()) {
		std::printf("No CS_*.spv files in %s\n", folder.string().c_str());
		return 1;
	}
	char self[MAX_PATH];
	GetModuleFileNameA(nullptr, self, MAX_PATH);

	std::ofstream results(folder / "amd_repro_results.txt");
	const auto report = [&](const std::string& line) {
		std::printf("%s\n", line.c_str());
		std::fflush(stdout);
		results << line << "\n";
		results.flush();
	};
	report("amd_repro: " + std::to_string(shaders.size()) + " compute shaders x " +
	       std::to_string(std::size(kVariants)) + " variants");
	for (const auto& shader: shaders) {
		for (const auto& variant: kVariants) {
			std::printf("%s [%s]\n", shader.filename().string().c_str(), variant.name);
			std::fflush(stdout);
			std::string command = std::string("\"") + self + "\" --child \"" + shader.string() + "\" " +
			                      std::to_string(variant.bits);
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
			std::snprintf(line, sizeof(line), "%-48s %-20s %s", shader.filename().string().c_str(),
			              variant.name, Describe(code).c_str());
			report(line);
		}
	}
	report("done");
	return 0;
}
