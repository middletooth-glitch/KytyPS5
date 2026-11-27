#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <csetjmp>
#include <string>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Resolves native descriptor sources, plans their scalar reads, and assigns dense resource bindings.
void TrackResources(Program& program, const Decoder::Program& decoded, const CFG::Graph& native_cfg,
                    ShaderStageInputInfo input);

// When set on the translating thread, a resource-tracking failure records its message here and
// longjmps to the buffer instead of terminating, so the caller can drop that shader's calls.
// Opt-in diagnostic path (KYTY_SKIP_UNSUPPORTED_SHADERS=1); objects on the unwound frames leak.
extern thread_local std::jmp_buf* g_resource_tracking_soft_fail;
extern thread_local std::string   g_resource_tracking_soft_fail_message;

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_ */
