const std = @import("std");
const builtin = @import("builtin");

/// Terminal helpers (ANSI colors + Windows VT enablement).
///
/// This module is intentionally lightweight and dependency-free.
pub const Ansi = struct {
    pub const reset = "\x1b[0m";
    pub const bold = "\x1b[1m";
    pub const dim = "\x1b[2m";

    pub const black = "\x1b[30m";
    pub const red = "\x1b[31m";
    pub const green = "\x1b[32m";
    pub const yellow = "\x1b[33m";
    pub const blue = "\x1b[34m";
    pub const magenta = "\x1b[35m";
    pub const cyan = "\x1b[36m";
    pub const white = "\x1b[37m";

    pub const bright_black = "\x1b[90m";
    pub const bright_red = "\x1b[91m";
    pub const bright_green = "\x1b[92m";
    pub const bright_yellow = "\x1b[93m";
    pub const bright_blue = "\x1b[94m";
    pub const bright_magenta = "\x1b[95m";
    pub const bright_cyan = "\x1b[96m";
    pub const bright_white = "\x1b[97m";
};

/// Attempts to enable ANSI color escape sequences on Windows.
///
/// Returns:
/// - `true` if ANSI sequences are expected to be supported (Windows VT enabled or non-Windows)
/// - `false` if the console mode could not be updated (caller should avoid emitting ANSI)
pub fn enableAnsiColors() bool {
    if (builtin.os.tag != .windows) return true;

    const kernel32 = std.os.windows.kernel32;
    const handle = kernel32.GetStdHandle(std.os.windows.STD_OUTPUT_HANDLE) orelse return false;
    if (handle == std.os.windows.INVALID_HANDLE_VALUE) return false;

    var mode: std.os.windows.DWORD = 0;
    if (kernel32.GetConsoleMode(handle, &mode) == 0) return false;

    // https://learn.microsoft.com/windows/console/setconsolemode
    const ENABLE_PROCESSED_OUTPUT: std.os.windows.DWORD = 0x0001;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: std.os.windows.DWORD = 0x0004;
    const DISABLE_NEWLINE_AUTO_RETURN: std.os.windows.DWORD = 0x0008;

    // Ensure VT is enabled for ANSI, but *clear* DISABLE_NEWLINE_AUTO_RETURN.
    // When that bit is set, '\n' moves down without returning to column 0,
    // which makes subsequent lines appear increasingly indented.
    const new_mode = (mode | ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING) & ~DISABLE_NEWLINE_AUTO_RETURN;
    if (kernel32.SetConsoleMode(handle, new_mode) == 0) return false;

    return true;
}
