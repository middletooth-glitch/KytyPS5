#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_TRACE3D_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_TRACE3D_H_

#include "common/logging/log.h"

#include <atomic>
#include <cstdio>

// Diagnostic trace of the steps around shaders that use 3D images, to find which one AMD's
// driver crashes after. Lines start with "trace3d:".
namespace Libs::Graphics::Trace3d {

// Set once a shader that uses a 3D image has been handed to the driver; submits are traced after.
inline std::atomic<bool>     g_seen {false};
inline std::atomic<uint32_t> g_submits {0};

inline bool TraceSubmit() {
	return g_seen.load(std::memory_order_relaxed) &&
	       g_submits.fetch_add(1, std::memory_order_relaxed) < 400u;
}

} // namespace Libs::Graphics::Trace3d

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define TRACE3D(...)                                                                               \
	do {                                                                                           \
		LOGF("trace3d: " __VA_ARGS__);                                                             \
		std::fflush(stdout);                                                                       \
	} while (false)

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_TRACE3D_H_ */
