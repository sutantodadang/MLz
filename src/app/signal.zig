const std = @import("std");
const builtin = @import("builtin");

const c = @cImport({
    @cInclude("signal.h");
});

/// Minimal Ctrl+C handling.
///
/// On Windows we use SetConsoleCtrlHandler.
/// On other platforms we fall back to a no-op (the app still works, but Ctrl+C
/// will terminate immediately).
///
/// This keeps the main loop responsive and allows `defer` cleanups + chat save.
pub const CtrlC = struct {
    installed: bool = false,

    pub fn init() !CtrlC {
        if (builtin.os.tag == .windows) {
            if (std.os.windows.kernel32.SetConsoleCtrlHandler(handler, std.os.windows.TRUE) == 0) {
                return error.CtrlCInitFailed;
            }
            return .{ .installed = true };
        }

        // POSIX fallback: install signal handlers. This allows the main loop to
        // observe `shouldExit()` and exit cleanly.
        _ = c.signal(c.SIGINT, posixHandler);
        _ = c.signal(c.SIGTERM, posixHandler);
        return .{ .installed = true };
    }

    pub fn deinit(self: CtrlC) void {
        if (builtin.os.tag == .windows and self.installed) {
            _ = std.os.windows.kernel32.SetConsoleCtrlHandler(handler, std.os.windows.FALSE);
        }
    }
};

var should_exit_flag = std.atomic.Value(bool).init(false);

pub fn shouldExit() bool {
    return should_exit_flag.load(.seq_cst);
}

fn handler(ctrl_type: std.os.windows.DWORD) callconv(.winapi) std.os.windows.BOOL {
    switch (ctrl_type) {
        std.os.windows.CTRL_C_EVENT,
        std.os.windows.CTRL_BREAK_EVENT,
        std.os.windows.CTRL_CLOSE_EVENT,
        => {
            should_exit_flag.store(true, .seq_cst);
            return std.os.windows.TRUE;
        },
        else => return std.os.windows.FALSE,
    }
}

fn posixHandler(sig: c_int) callconv(.c) void {
    _ = sig;
    should_exit_flag.store(true, .seq_cst);
}
