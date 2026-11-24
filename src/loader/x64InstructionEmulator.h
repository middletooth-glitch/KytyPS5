#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <cstddef>
#include <cstdint>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool TryEmulate(void* native_context);
// Completes a faulting load into EAX as a zero read and resumes after it.
void SkipLoad(void* native_context, size_t length);
// An SRT walker load (mov eax, [r10]) completed with `value`: the address it reads, then the
// result written and the instruction skipped.
uint64_t SrtWalkerLoadAddress(void* native_context);
void     CompleteSrtWalkerLoad(void* native_context, size_t length, uint32_t value);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
