#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

// Engine console variables for the game's own command-line file: KYTY_GAME_CONVARS="name=value;name=value"
// gives a read-only open of the PackageCmdLineArgs.txt the game ships a copy with a "+name=value" line per
// entry added (Demon's Souls passes every line of that file to its console at boot), so engine settings
// change without touching the game's files.
namespace Loader::GameArgs {

// Console variables a title gets when KYTY_GAME_CONVARS is not set (set it, even empty, to override):
// Demon's Souls (PPSA01341) runs without its XPR index-buffer culling pass (doIndexBufferCulling=false),
// ~2,800 compute dispatches a frame whose results the CPU reads back. Call before the game starts.
void SetTitleDefaults(const std::string& title_id);

// The host file a read-only open of `real` should open instead (empty: `real` itself).
std::filesystem::path RedirectRead(const std::filesystem::path& real);
// A stat of `real`: the size of the copy a read would get (nothing: its own).
std::optional<uint64_t> RedirectedSize(const std::filesystem::path& real);

} // namespace Loader::GameArgs
