const std = @import("std");

/// Chat domain model + persistence utilities.
///
/// This file intentionally has no dependency on llama.cpp bindings; it is pure Zig.
pub const Role = enum {
    system,
    user,
    assistant,
};

pub const Message = struct {
    role: Role,
    /// UTF-8, null-terminated for llama.cpp chat template APIs.
    /// The slice length excludes the sentinel terminator.
    content: [:0]u8,
};

pub const ChatError = error{
    InvalidFormat,
    InvalidRole,
    NulByteInString,
};

pub fn dupeZ(allocator: std.mem.Allocator, s: []const u8) ![:0]u8 {
    if (std.mem.indexOfScalar(u8, s, 0) != null) return ChatError.NulByteInString;
    const out = try allocator.alloc(u8, s.len + 1);
    @memcpy(out[0..s.len], s);
    out[s.len] = 0;
    return out[0..s.len :0];
}

pub fn deinitMessages(allocator: std.mem.Allocator, msgs: *std.ArrayList(Message)) void {
    for (msgs.items) |m| allocator.free(m.content);
    msgs.deinit(allocator);
}

pub fn roleToString(role: Role) []const u8 {
    return switch (role) {
        .system => "system",
        .user => "user",
        .assistant => "assistant",
    };
}

pub fn roleFromString(s: []const u8) ChatError!Role {
    if (std.mem.eql(u8, s, "system")) return .system;
    if (std.mem.eql(u8, s, "user")) return .user;
    if (std.mem.eql(u8, s, "assistant")) return .assistant;
    return ChatError.InvalidRole;
}

/// Ensures the chat begins with a system prompt.
/// If an existing system message exists at index 0, it is replaced.
/// Otherwise, it is prepended.
pub fn setOrPrependSystemPrompt(
    allocator: std.mem.Allocator,
    msgs: *std.ArrayList(Message),
    prompt: []const u8,
) !void {
    const sys_z = try dupeZ(allocator, prompt);
    errdefer allocator.free(sys_z);

    if (msgs.items.len > 0 and msgs.items[0].role == .system) {
        allocator.free(msgs.items[0].content);
        msgs.items[0].content = sys_z;
        return;
    }

    // Prepend by inserting at index 0 (keep order).
    try msgs.insert(allocator, 0, .{ .role = .system, .content = sys_z });
}

/// Removes the oldest non-system messages to reduce prompt size.
/// This is a sliding window: it preserves system prompt (if any) and recent history.
///
/// Returns true if any messages were dropped.
pub fn dropOldestNonSystem(msgs: *std.ArrayList(Message), allocator: std.mem.Allocator) bool {
    const start_index: usize = if (msgs.items.len > 0 and msgs.items[0].role == .system) 1 else 0;
    if (msgs.items.len <= start_index) return false;

    // Drop one message at a time; caller can repeat until it fits.
    const removed = msgs.orderedRemove(start_index);
    allocator.free(removed.content);
    return true;
}

/// Clears all conversation messages while preserving the system prompt (if present).
///
/// This is used to implement `/clear` and `/reset` without reloading the model.
pub fn clearKeepSystem(allocator: std.mem.Allocator, msgs: *std.ArrayList(Message)) void {
    const keep_system = msgs.items.len > 0 and msgs.items[0].role == .system;
    const keep_len: usize = if (keep_system) 1 else 0;

    while (msgs.items.len > keep_len) {
        const removed = msgs.pop().?;
        allocator.free(removed.content);
    }
}

const JsonMsg = struct {
    role: []const u8,
    content: []const u8,
};

/// Loads messages from a JSON file.
/// Format: [{"role":"system|user|assistant","content":"..."}, ...]
pub fn loadJson(allocator: std.mem.Allocator, path: []const u8) !std.ArrayList(Message) {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const data = try file.readToEndAlloc(allocator, 64 * 1024 * 1024);
    defer allocator.free(data);

    var out: std.ArrayList(Message) = .empty;
    errdefer deinitMessages(allocator, &out);

    var parsed = try std.json.parseFromSlice([]JsonMsg, allocator, data, .{});
    defer parsed.deinit();

    for (parsed.value) |jm| {
        const role = try roleFromString(jm.role);
        const content_z = try dupeZ(allocator, jm.content);
        errdefer allocator.free(content_z);

        try out.append(allocator, .{ .role = role, .content = content_z });
    }

    return out;
}

/// Saves messages to a JSON file (atomic write).
pub fn saveJson(allocator: std.mem.Allocator, path: []const u8, msgs: []const Message) !void {
    _ = allocator;
    var write_buffer: [16 * 1024]u8 = undefined;
    var atomic = try std.fs.cwd().atomicFile(path, .{ .write_buffer = write_buffer[0..] });
    defer atomic.deinit();

    var jw: std.json.Stringify = .{
        .writer = &atomic.file_writer.interface,
        .options = .{ .whitespace = .indent_2 },
    };

    try jw.beginArray();
    for (msgs) |m| {
        try jw.beginObject();
        try jw.objectField("role");
        try jw.write(roleToString(m.role));
        try jw.objectField("content");
        try jw.write(m.content[0..m.content.len]);
        try jw.endObject();
    }
    try jw.endArray();

    try atomic.finish();
}
