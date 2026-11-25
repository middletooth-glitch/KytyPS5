#ifndef GRAPHICS_HOST_GPU_RENDERER_WAIT_PROFILE_H_
#define GRAPHICS_HOST_GPU_RENDERER_WAIT_PROFILE_H_

// Diagnostic: KYTY_WAIT_PROFILE=1 attributes every blocking GPU timeline wait
// (MasterSemaphore::Wait) to the nested WaitProfile::Tag scopes active on the waiting thread and
// logs the totals every 10 seconds, so the waits that cost frame time can be found.

#include "common/logging/log.h"

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>

namespace Libs::Graphics::WaitProfile {

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_WAIT_PROFILE");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

struct Tag;
inline thread_local const Tag* t_current = nullptr;
// Guest hash of the shader whose program was looked up last on this thread (attribution only).
inline thread_local uint64_t t_last_shader_hash = 0;

struct Tag {
	const char* name;
	const Tag*  parent;
	explicit Tag(const char* tag_name): name(tag_name), parent(t_current) { t_current = this; }
	~Tag() { t_current = parent; }
	Tag(const Tag&)            = delete;
	Tag& operator=(const Tag&) = delete;
};

struct Totals {
	uint64_t count = 0;
	uint64_t ns    = 0;
	uint64_t bytes = 0;
};

inline void Add(std::string key, uint64_t ns, uint64_t bytes);

// A blocking wait of ns nanoseconds, attributed to the active tags.
inline void Record(uint64_t ns) {
	if (!Enabled()) {
		return;
	}
	std::string key;
	{
		std::vector<const char*> names;
		for (const Tag* tag = t_current; tag != nullptr; tag = tag->parent) {
			names.push_back(tag->name);
		}
		if (names.empty()) {
			key = "(untagged)";
		}
		for (auto it = names.rbegin(); it != names.rend(); ++it) {
			if (!key.empty()) key += " > ";
			key += *it;
		}
	}
	Add(std::move(key), ns, 0);
}

// An event that is not a wait (counted, with bytes), such as a skipped readback.
inline void Event(const std::string& name, uint64_t bytes = 0) {
	if (!Enabled()) {
		return;
	}
	Add("[event] " + name, 0, bytes);
}

inline void Add(std::string key, uint64_t ns, uint64_t bytes) {
	static std::mutex                              mutex;
	static std::unordered_map<std::string, Totals> totals;
	static auto next_report = std::chrono::steady_clock::now() + std::chrono::seconds(10);
	static auto window_start = std::chrono::steady_clock::now();
	std::lock_guard lock(mutex);
	auto& entry = totals[key];
	entry.count++;
	entry.ns += ns;
	entry.bytes += bytes;
	const auto now = std::chrono::steady_clock::now();
	if (now < next_report) {
		return;
	}
	const double seconds =
	    std::chrono::duration<double>(now - window_start).count();
	std::vector<std::pair<std::string, Totals>> sorted(totals.begin(), totals.end());
	std::ranges::sort(sorted, [](const auto& a, const auto& b) {
		return a.second.ns != b.second.ns ? a.second.ns > b.second.ns
		                                  : a.second.count > b.second.count;
	});
	LOGF("--- GPU wait profile (%.1f s) ---\n", seconds);
	if (sorted.size() > 100) {
		sorted.resize(100);
	}
	for (const auto& [name, value]: sorted) {
		LOGF("  %8.1f ms/s %8.1f /s %8.2f MB/s  %s\n",
		     static_cast<double>(value.ns) / 1e6 / seconds,
		     static_cast<double>(value.count) / seconds,
		     static_cast<double>(value.bytes) / 1048576.0 / seconds, name.c_str());
	}
	totals.clear();
	window_start = now;
	next_report  = now + std::chrono::seconds(10);
}

} // namespace Libs::Graphics::WaitProfile

#endif /* GRAPHICS_HOST_GPU_RENDERER_WAIT_PROFILE_H_ */
