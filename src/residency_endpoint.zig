//! OpenAI-compatible HTTP endpoint for the bounded-residency completion
//! service.
//!
//! Route: `POST /v1/residency/completions` (routed by `src/server.zig`).
//! Every request executes `ResidencyService` against the server's startup
//! model with an explicit mapped-weight budget: weight tiles are mapped,
//! pinned, and unmapped per request, so resident weight memory stays bounded
//! regardless of model size.
//!
//! The handler is serialized by a mutex because the underlying service uses a
//! single-owner executor; concurrent requests are answered sequentially.
//! `stream: true` produces OpenAI-style SSE chunks; otherwise a single JSON
//! completion object is returned.

const std = @import("std");
const service_mod = @import("residency_service.zig");

const ResidencyService = service_mod.ResidencyService;

pub const Error = error{
    BadRequest,
    Internal,
    OutOfMemory,
};

/// Shared, lazily-initialized residency service bound to the server's startup
/// model path. Guarded by `mutex`; `service` stays `null` until the first
/// request arrives, so a server that never receives a residency request never
/// pays for the service. The mutex serializes requests because the service
/// uses a single-owner executor.
pub const ResidencyEndpoint = struct {
    allocator: std.mem.Allocator,
    model_path: []const u8,
    /// Mapped-weight budget in bytes.
    budget_bytes: usize,
    /// Optional non-weight state budget in bytes (0 = unlimited).
    state_cache_bytes: usize = 0,
    state_workspace_bytes: usize = 0,

    mutex: std.Thread.Mutex = .{},
    service: ?*ResidencyService = null,

    /// Initializes the endpoint. `model_path` is duplicated.
    pub fn init(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        budget_bytes: usize,
    ) !ResidencyEndpoint {
        return .{
            .allocator = allocator,
            .model_path = try allocator.dupe(u8, model_path),
            .budget_bytes = budget_bytes,
        };
    }

    pub fn deinit(self: *ResidencyEndpoint) void {
        self.reset();
        self.allocator.free(self.model_path);
    }

    /// Opens the service on first use. Caller must hold `mutex`.
    fn ensureService(self: *ResidencyEndpoint) Error!*ResidencyService {
        if (self.service) |s| return s;
        const s = try self.allocator.create(ResidencyService);
        errdefer self.allocator.destroy(s);
        s.* = ResidencyService.open(self.allocator, self.model_path, .{
            .budget_bytes = self.budget_bytes,
        }) catch |err| switch (err) {
            error.OutOfMemory => |e| return e,
            else => return Error.Internal,
        };
        self.service = s;
        return s;
    }

    /// Closes the service (if open) so the next request reopens it.
    pub fn reset(self: *ResidencyEndpoint) void {
        if (self.service) |s| {
            s.close();
            self.allocator.destroy(s);
            self.service = null;
        }
    }
};

const max_id_len = 64;

/// Reads a string field from a JSON object.
fn getString(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

/// Reads an integer field from a JSON object.
fn getInt(obj: std.json.ObjectMap, key: []const u8) ?i64 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |i| i,
        .float => |f| if (f >= 0) @as(i64, @intFromFloat(f)) else null,
        else => null,
    };
}

fn getBool(obj: std.json.ObjectMap, key: []const u8) ?bool {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .bool => |b| b,
        else => null,
    };
}

fn getFloat(obj: std.json.ObjectMap, key: []const u8) ?f64 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

/// Escapes a string into a JSON string literal (with surrounding quotes).
fn writeJsonString(w: anytype, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |ch| {
        switch (ch) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            else => {
                if (ch < 0x20) {
                    try w.print("\\u{x:0>4}", .{ch});
                } else {
                    try w.writeByte(ch);
                }
            },
        }
    }
    try w.writeByte('"');
}

/// Synchronous token sink bridging `ResidencyService` streaming into
/// OpenAI-style SSE chunks.
const SseBridge = struct {
    stream: std.net.Stream,
    id: []const u8,
    model: []const u8,

    fn onToken(context: ?*anyopaque, token: usize, piece: []const u8) void {
        const self: *SseBridge = @ptrCast(@alignCast(context.?));
        self.emit(token, piece) catch {};
    }

    fn emit(self: *SseBridge, token: usize, piece: []const u8) !void {
        var buf: [2048]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        const w = fbs.writer();

        try w.writeAll("data: {\"id\":\"");
        try w.writeAll(self.id);
        try w.writeAll("\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"");
        try w.writeAll(self.model);
        try w.writeAll("\",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
        try writeJsonString(w, piece);
        try w.writeAll("},\"finish_reason\":null}]}\n\n");
        _ = token;

        try self.stream.writeAll(fbs.getWritten());
    }
};

fn writeJsonError(
    stream: std.net.Stream,
    status: u16,
    reason: []const u8,
    code: []const u8,
    message: []const u8,
) !void {
    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    const w = fbs.writer();

    try w.print("HTTP/1.1 {d} {s}\r\nContent-Type: application/json\r\nContent-Length: ", .{ status, reason });
    const header_end = fbs.getWritten().len;
    try w.writeAll("\r\n\r\n{\"error\":{\"message\":");
    try writeJsonString(w, message);
    try w.writeAll(",\"type\":\"invalid_request_error\",\"code\":");
    try writeJsonString(w, code);
    try w.writeAll("}}");

    // Patch the content-length in-place.
    const body = fbs.getWritten()[header_end..];
    var cl_buf: [16]u8 = undefined;
    const cl = std.fmt.bufPrint(&cl_buf, "{d}", .{body.len - 2}) catch "?";
    var out: [512]u8 = undefined;
    var out_fbs = std.io.fixedBufferStream(&out);
    const ow = out_fbs.writer();
    try ow.print("HTTP/1.1 {d} {s}\r\nContent-Type: application/json\r\nContent-Length: {s}\r\n\r\n", .{ status, reason, cl });
    try ow.writeAll(body);

    try stream.writeAll(out_fbs.getWritten());
}

/// Handles one `POST /v1/residency/completions` request. Writes a JSON
/// completion object or an SSE stream to `stream`. Returns an error category
/// only for pre-response failures; once writing begins, errors are answered
/// inline.
pub fn handle(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    endpoint: *ResidencyEndpoint,
    body: []const u8,
) anyerror!void {
    endpoint.mutex.lock();
    defer endpoint.mutex.unlock();

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        else => {
            writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "malformed JSON body") catch {};
            return;
        },
    };
    defer parsed.deinit();

    const obj = switch (parsed.value) {
        .object => |o| o,
        else => {
            writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "body must be a JSON object") catch {};
            return;
        },
    };

    // Input modes: `messages` (chat, rendered via the model's jinja template)
    // or `prompt` (raw completion text). Exactly one is required.
    var prompt_tokens: ?[]usize = null;
    defer if (prompt_tokens) |t| allocator.free(t);
    if (obj.get("messages")) |mv| {
        switch (mv) {
            .array => |arr| {
                if (arr.items.len == 0) {
                    writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "messages must not be empty") catch {};
                    return;
                }
                var messages = std.ArrayList(service_mod.ResidencyService.ChatMessage).empty;
                defer messages.deinit(allocator);
                for (arr.items) |item| {
                    const mobj = switch (item) {
                        .object => |o| o,
                        else => {
                            writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "messages entries must be objects") catch {};
                            return;
                        },
                    };
                    const role = getString(mobj, "role") orelse {
                        writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "message missing role") catch {};
                        return;
                    };
                    const content = getString(mobj, "content") orelse {
                        writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "message missing content") catch {};
                        return;
                    };
                    messages.append(allocator, .{ .role = role, .content = content }) catch |e| return e;
                }
                const service = endpoint.ensureService() catch |err| switch (err) {
                    error.OutOfMemory => |e| return e,
                    else => {
                        writeJsonError(stream, 503, "Service Unavailable", "server_error", "failed to open the residency service") catch {};
                        return;
                    },
                };
                const tokenized = service.applyChatTemplate(messages.items) catch |err| switch (err) {
                    error.OutOfMemory => |e| return e,
                    else => {
                        writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "failed to apply chat template") catch {};
                        return;
                    },
                };
                prompt_tokens = tokenized.tokens;
            },
            else => {
                writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "messages must be an array") catch {};
                return;
            },
        }
    } else if (obj.get("prompt")) |pv| {
        const prompt = switch (pv) {
            .string => |s| s,
            else => {
                writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "prompt must be a string") catch {};
                return;
            },
        };
        if (prompt.len == 0) {
            writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "prompt must not be empty") catch {};
            return;
        }
        const service = endpoint.ensureService() catch |err| switch (err) {
            error.OutOfMemory => |e| return e,
            else => {
                writeJsonError(stream, 503, "Service Unavailable", "server_error", "failed to open the residency service") catch {};
                return;
            },
        };
        // Tokenize the raw prompt text against the service vocabulary.
        // add_bos is disabled to match the raw-completion semantics of the CLI
        // service.
        const tokenized = service.tokenize(prompt, false) catch |err| switch (err) {
            error.OutOfMemory => |e| return e,
            else => {
                writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "failed to tokenize prompt") catch {};
                return;
            },
        };
        prompt_tokens = tokenized.tokens;
    } else {
        writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "missing required field: prompt or messages") catch {};
        return;
    }

    const max_tokens: usize = blk: {
        const v = getInt(obj, "max_tokens") orelse break :blk 32;
        if (v < 0 or v > 4096) {
            writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "max_tokens out of range") catch {};
            return;
        }
        break :blk @intCast(v);
    };

    const streaming = getBool(obj, "stream") orelse false;

    var sampling: service_mod.SamplingStrategy = .greedy;
    var temperature: f32 = 1.0;
    var top_k: usize = 0;
    if (getFloat(obj, "temperature")) |t| {
        sampling = .temperature;
        if (t <= 0.0 or !std.math.isFinite(t) or t > 100.0) {
            writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "temperature out of range") catch {};
            return;
        }
        temperature = @floatCast(t);
    }
    if (getInt(obj, "top_k")) |k| {
        if (k < 0 or k > 1024) {
            writeJsonError(stream, 400, "Bad Request", "invalid_request_error", "top_k out of range") catch {};
            return;
        }
        top_k = @intCast(k);
    }

    const service = endpoint.ensureService() catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        else => {
            writeJsonError(stream, 503, "Service Unavailable", "server_error", "failed to open the residency service") catch {};
            return;
        },
    };

    // Request id: short hex from the timestamp.
    var id_buf: [max_id_len]u8 = undefined;
    const id = std.fmt.bufPrint(&id_buf, "cmpl-{x}", .{std.time.nanoTimestamp()}) catch "cmpl-0";

    var options = service_mod.ResidencyService.CompletionOptions{
        .max_tokens = max_tokens,
        .sampling = sampling,
        .temperature = temperature,
        .top_k = top_k,
        .prompt_tokens = prompt_tokens.?,
    };
    if (endpoint.state_cache_bytes != 0 or endpoint.state_workspace_bytes != 0) {
        options.state_budget = .{
            .cache_bytes = endpoint.state_cache_bytes,
            .workspace_bytes = endpoint.state_workspace_bytes,
        };
    }

    var sink_bridge = SseBridge{
        .stream = stream,
        .id = id,
        .model = "residency",
    };
    if (streaming) {
        options.token_sink = .{
            .context = &sink_bridge,
            .callback = SseBridge.onToken,
        };
        try stream.writeAll("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");
    }

    const result = service.complete(.{ .budget_bytes = endpoint.budget_bytes }, options) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        else => {
            if (streaming) {
                try stream.writeAll("data: {\"error\":{\"message\":\"completion failed\",\"type\":\"server_error\"}}\n\n");
                try stream.writeAll("data: [DONE]\n\n");
            } else {
                var msg_buf: [128]u8 = undefined;
                const msg = std.fmt.bufPrint(&msg_buf, "completion failed: {s}", .{@errorName(err)}) catch "completion failed";
                writeJsonError(stream, 500, "Internal Server Error", "server_error", msg) catch {};
            }
            return;
        },
    };
    defer {
        allocator.free(result.text);
        allocator.free(result.tokens);
    }

    if (streaming) {
        var buf: [512]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        const w = fbs.writer();
        try w.writeAll("data: {\"id\":\"");
        try w.writeAll(id);
        try w.writeAll("\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"residency\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n");
        try w.writeAll("data: [DONE]\n\n");
        try stream.writeAll(fbs.getWritten());
        return;
    }

    // Non-streaming: single JSON completion object.
    var body_buf: std.ArrayList(u8) = .empty;
    defer body_buf.deinit(allocator);
    const w = body_buf.writer(allocator);

    try w.writeAll("{\"id\":");
    try writeJsonString(w, id);
    try w.writeAll(",\"object\":\"text_completion\",\"created\":");
    try w.print("{d}", .{std.time.timestamp()});
    try w.writeAll(",\"model\":\"residency\",\"choices\":[{\"index\":0,\"text\":");
    try writeJsonString(w, result.text);
    try w.writeAll(",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":");
    try w.print("{d}", .{result.prompt_tokens});
    try w.writeAll(",\"completion_tokens\":");
    try w.print("{d}", .{result.tokens.len});
    try w.writeAll(",\"total_tokens\":");
    try w.print("{d}", .{result.prompt_tokens + result.tokens.len});
    try w.writeAll("},\"residency\":{\"weight_budget_bytes\":");
    try w.print("{d}", .{result.weight_budget_bytes});
    try w.writeAll(",\"peak_mapped_weight_bytes\":");
    try w.print("{d}", .{result.peak_mapped_weight_bytes});
    try w.writeAll(",\"faults\":");
    try w.print("{d}", .{result.faults});
    try w.writeAll(",\"evictions\":");
    try w.print("{d}", .{result.evictions});
    try w.writeAll(",\"kv_cache_bytes\":");
    try w.print("{d}", .{result.kv_cache_bytes});
    try w.writeAll("}}");

    var hdr_buf: [96]u8 = undefined;
    const headers = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n", .{body_buf.items.len}) catch unreachable;
    try stream.writeAll(headers);
    try stream.writeAll(body_buf.items);
}

test "writeJsonString escapes quotes and control characters" {
    var buf: [128]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try writeJsonString(fbs.writer(), "a\"b\\c\nd\te");
    try std.testing.expectEqualStrings("\"a\\\"b\\\\c\\nd\\te\"", fbs.getWritten());
}

test "writeJsonString passes through printable utf-8" {
    var buf: [64]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try writeJsonString(fbs.writer(), "héllo — ok");
    try std.testing.expectEqualStrings("\"héllo — ok\"", fbs.getWritten());
}
