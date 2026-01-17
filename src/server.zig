const std = @import("std");

const llama_cpp = @import("llama_cpp.zig");
const chat = @import("chat.zig");
const signal = @import("signal.zig");
const inference = @import("inference.zig");
const openai = @import("openai.zig");

/// Configuration for the HTTP server.
pub const ServerConfig = struct {
    /// Host address to bind to (e.g., "127.0.0.1").
    host: []const u8 = "127.0.0.1",
    /// Port to listen on (e.g., 8080).
    port: u16 = 8080,

    /// If set, require `Authorization: Bearer <api_key>`.
    api_key: ?[]const u8 = null,

    /// Maximum header bytes to read.
    max_header_bytes: usize = 64 * 1024,

    /// Maximum request body bytes to read.
    max_body_bytes: usize = 2 * 1024 * 1024,

    /// If true, logs basic request lines.
    log_requests: bool = true,
};

/// Configuration for the inference engine.
pub const EngineConfig = struct {
    // Model/context
    /// Context window size (tokens).
    n_ctx: u32 = 4096,
    /// Number of layers to offload to GPU.
    n_gpu_layers: i32 = 999,
    /// Number of threads to use for generation (null = auto-detect).
    threads: ?i32 = null,

    // Sampler defaults
    /// Sampling temperature (higher = more random).
    temp: f32 = 0.8,
    /// Top-K sampling.
    top_k: i32 = 40,
    /// Top-P (nucleus) sampling.
    top_p: f32 = 0.95,
    /// Min-P sampling.
    min_p: f32 = 0.05,
    /// Random seed.
    seed: u32 = 42,

    // Optional grammar
    /// Path to a GBNF grammar file.
    grammar_path: ?[]const u8 = null,
    /// Root rule name for the grammar.
    grammar_root: []const u8 = "root",
};

const Header = struct {
    name_lc: []const u8,
    value: []const u8,
};

/// Represents an incoming HTTP request.
const HttpRequest = struct {
    method: []const u8,
    path: []const u8,
    headers: []Header,
    body: []const u8,
};

const HttpResponse = struct {
    status: u16,
    reason: []const u8,
    content_type: []const u8,
    body: []const u8,
};

const WsOpcode = enum(u4) {
    continuation = 0x0,
    text = 0x1,
    binary = 0x2,
    close = 0x8,
    ping = 0x9,
    pong = 0xA,
};

const WsFrame = struct {
    opcode: WsOpcode,
    payload: []u8,
    fin: bool,

    pub fn deinit(self: WsFrame, allocator: std.mem.Allocator) void {
        allocator.free(self.payload);
    }
};

/// Starts the HTTP server with the given model and configuration.
/// This function blocks until the server is stopped via signal (Ctrl+C).
pub fn run(allocator: std.mem.Allocator, model_path: []const u8, cfg: ServerConfig, engine_cfg: EngineConfig) !void {
    var backend = llama_cpp.Backend.init();
    defer backend.deinit();

    const ctrlc = try signal.CtrlC.init();
    defer ctrlc.deinit();

    var engine = try Engine.init(allocator, model_path, engine_cfg);
    defer engine.deinit(allocator);

    const addr = try resolveListenAddress(allocator, cfg.host, cfg.port);
    defer allocator.free(addr.host);

    var server = try addr.address.listen(.{
        .reuse_address = true,
        .force_nonblocking = false,
        .kernel_backlog = 128,
    });
    defer server.deinit();

    std.log.info("server listening on {s}:{d}", .{ cfg.host, cfg.port });

    while (!signal.shouldExit()) {
        const conn = server.accept() catch |err| {
            if (signal.shouldExit()) break;
            return err;
        };

        const Handler = struct {
            allocator: std.mem.Allocator,
            stream: std.net.Stream,
            engine: *Engine,
            cfg: ServerConfig,

            fn run(self: @This()) void {
                handleConnection(self.allocator, self.stream, self.engine, self.cfg) catch |err| {
                    std.log.err("connection error: {any}", .{err});
                };
                self.stream.close();
            }
        };

        const handler = Handler{
            .allocator = allocator,
            .stream = conn.stream,
            .engine = &engine,
            .cfg = cfg,
        };

        const thread = std.Thread.spawn(.{}, Handler.run, .{handler}) catch |err| {
            std.log.err("failed to spawn thread: {any}", .{err});
            conn.stream.close();
            continue;
        };
        thread.detach();
    }
}

const ResolvedAddress = struct {
    host: []u8,
    address: std.net.Address,
};

fn resolveListenAddress(allocator: std.mem.Allocator, host: []const u8, port: u16) !ResolvedAddress {
    // Fast path for IP literals.
    if (std.net.Address.parseIp(host, port)) |address| {
        return .{ .host = try allocator.dupe(u8, host), .address = address };
    } else |_| {}

    // Fallback to DNS resolution.
    var list = try std.net.getAddressList(allocator, host, port);
    defer list.deinit();

    if (list.addrs.len == 0) return error.AddressNotResolved;
    return .{ .host = try allocator.dupe(u8, host), .address = list.addrs[0] };
}

fn handleConnection(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    engine: *Engine,
    cfg: ServerConfig,
) !void {
    var req_buf: std.ArrayList(u8) = .empty;
    defer req_buf.deinit(allocator);

    const req = readHttpRequest(allocator, stream, &req_buf, cfg.max_header_bytes, cfg.max_body_bytes) catch |err| {
        std.log.err("request reading failed: {any}", .{err});
        return err;
    };
    defer freeHeaders(allocator, req.headers);

    if (cfg.log_requests) {
        std.log.info("{s} {s}", .{ req.method, req.path });
    }

    if (cfg.api_key) |key| {
        if (!authorized(req.headers, key)) {
            try writeJsonError(allocator, stream, 401, "Unauthorized", "invalid_api_key", "missing or invalid Authorization header");
            return;
        }
    }

    if (isWebSocketUpgrade(req.headers) and std.mem.eql(u8, req.path, "/v1/chat/completions/ws")) {
        try handleWebSocket(allocator, stream, req.headers, engine);
        return;
    }

    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/health")) {
        const body = "{\"ok\":true}";
        try writeResponse(stream, .{ .status = 200, .reason = "OK", .content_type = "application/json", .body = body });
        return;
    }

    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/models")) {
        try handleModels(allocator, stream, engine);
        return;
    }

    if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/v1/chat/completions")) {
        try handleChatCompletions(allocator, stream, engine, req.body);
        return;
    }

    try writeJsonError(allocator, stream, 404, "Not Found", "not_found", "route not found");
}

fn authorized(headers: []const Header, api_key: []const u8) bool {
    const auth = headerValue(headers, "authorization") orelse return false;
    const prefix = "Bearer ";
    if (!std.mem.startsWith(u8, auth, prefix)) return false;
    const token = auth[prefix.len..];
    return std.mem.eql(u8, token, api_key);
}

fn handleModels(allocator: std.mem.Allocator, stream: std.net.Stream, engine: *Engine) !void {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    const model_id = engine.modelId();
    const ModelEntry = struct {
        id: []const u8,
        object: []const u8,
        owned_by: []const u8,
    };
    const ModelsResponse = struct {
        object: []const u8,
        data: []const ModelEntry,
    };

    var data = [_]ModelEntry{.{ .id = model_id, .object = "model", .owned_by = "mlz" }};
    const payload: ModelsResponse = .{ .object = "list", .data = data[0..] };

    try openai.writeJson(buf.writer(allocator), payload);

    try writeResponse(stream, .{
        .status = 200,
        .reason = "OK",
        .content_type = "application/json",
        .body = buf.items,
    });
}

fn handleChatCompletions(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    engine: *Engine,
    body: []const u8,
) !void {
    var parsed = openai.parseChatCompletionRequest(allocator, body) catch |err| {
        const msg = switch (err) {
            openai.ParseError.InvalidJson => "invalid JSON",
            openai.ParseError.MissingMessages => "messages must be non-empty",
        };
        try writeJsonError(allocator, stream, 400, "Bad Request", "invalid_request_error", msg);
        return;
    };
    defer parsed.deinit();

    const req = parsed.value;
    const stream_resp: bool = req.stream orelse false;

    if (!stream_resp) {
        var resp = try engine.complete(allocator, req, null, null);
        defer resp.deinit(allocator);

        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(allocator);
        try openai.writeJson(buf.writer(allocator), resp.value);

        try writeResponse(stream, .{
            .status = 200,
            .reason = "OK",
            .content_type = "application/json",
            .body = buf.items,
        });
        return;
    }

    // Streaming response using SSE (OpenAI-compatible).
    try writeSseHeaders(stream);

    var sse = SseSink.init(stream, allocator, engine.modelId());
    defer sse.deinit();

    // Initial role chunk.
    const id = try engine.nextIdAlloc(allocator);
    defer allocator.free(id);
    try sse.sendRole(id, "assistant");

    var resp = try engine.complete(allocator, req, sse.tokenSink(), id);
    defer resp.deinit(allocator);

    // Final finish_reason chunk.
    try sse.sendFinish(resp.finishReasonString());
    try sse.done();
}

fn writeSseHeaders(stream: std.net.Stream) !void {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    const writer = fbs.writer();

    try writer.writeAll("HTTP/1.1 200 OK\r\n");
    try writer.writeAll("Content-Type: text/event-stream\r\n");
    try writer.writeAll("Cache-Control: no-cache\r\n");
    try writer.writeAll("Connection: close\r\n");
    try writer.writeAll("\r\n");

    try stream.writeAll(fbs.getWritten());
}

const SseSink = struct {
    stream: std.net.Stream,
    allocator: std.mem.Allocator,
    id: []const u8,
    created: i64,
    model: []const u8,

    id_owned: ?[]u8 = null,

    pub fn init(stream: std.net.Stream, allocator: std.mem.Allocator, model: []const u8) SseSink {
        return .{
            .stream = stream,
            .allocator = allocator,
            .id = "",
            .created = std.time.timestamp(),
            .model = model,
            .id_owned = null,
        };
    }

    pub fn deinit(self: *SseSink) void {
        if (self.id_owned) |owned| self.allocator.free(owned);
        self.id_owned = null;
    }

    pub fn tokenSink(self: *SseSink) inference.TokenSink {
        return .{ .ctx = self, .writeFn = sseWriteToken, .flushFn = sseFlush };
    }

    pub fn sendRole(self: *SseSink, id: []const u8, role: []const u8) !void {
        if (self.id_owned) |owned| self.allocator.free(owned);
        self.id_owned = try self.allocator.dupe(u8, id);
        self.id = self.id_owned.?;
        self.created = std.time.timestamp();
        const chunk = openai.ChatCompletionChunk{
            .id = self.id,
            .object = "chat.completion.chunk",
            .created = self.created,
            .model = self.model,
            .choices = &[_]openai.ChatCompletionChunkChoice{.{
                .index = 0,
                .delta = .{ .role = role, .content = null },
                .finish_reason = null,
            }},
        };
        try self.sendChunk(chunk);
    }

    pub fn sendFinish(self: *SseSink, finish_reason: []const u8) !void {
        const chunk = openai.ChatCompletionChunk{
            .id = self.id,
            .object = "chat.completion.chunk",
            .created = self.created,
            .model = self.model,
            .choices = &[_]openai.ChatCompletionChunkChoice{.{
                .index = 0,
                .delta = .{ .role = null, .content = null },
                .finish_reason = finish_reason,
            }},
        };
        try self.sendChunk(chunk);
    }

    pub fn done(self: *SseSink) !void {
        try self.stream.writeAll("data: [DONE]\n\n");
    }

    fn sendChunk(self: *SseSink, chunk: openai.ChatCompletionChunk) !void {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(self.allocator);

        try buf.appendSlice(self.allocator, "data: ");
        try openai.writeJson(buf.writer(self.allocator), chunk);
        try buf.appendSlice(self.allocator, "\n\n");

        try self.stream.writeAll(buf.items);
    }

    fn sseFlush(ctx: *anyopaque) anyerror!void {
        // No-op since we write directly to stream
        _ = ctx;
    }

    fn sseWriteToken(ctx: *anyopaque, bytes: []const u8) anyerror!void {
        const self: *SseSink = @ptrCast(@alignCast(ctx));
        const chunk = openai.ChatCompletionChunk{
            .id = self.id,
            .object = "chat.completion.chunk",
            .created = self.created,
            .model = self.model,
            .choices = &[_]openai.ChatCompletionChunkChoice{.{
                .index = 0,
                .delta = .{ .role = null, .content = bytes },
                .finish_reason = null,
            }},
        };
        try self.sendChunk(chunk);
    }
};

fn writeJsonError(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    status: u16,
    reason: []const u8,
    err_type: []const u8,
    msg: []const u8,
) !void {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    const payload = openai.ErrorResponse{ .@"error" = .{ .message = msg, .type = err_type } };
    try openai.writeJson(buf.writer(allocator), payload);

    try writeResponse(stream, .{
        .status = status,
        .reason = reason,
        .content_type = "application/json",
        .body = buf.items,
    });
}

fn writeResponse(stream: std.net.Stream, resp: HttpResponse) !void {
    var buf: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    const writer = fbs.writer();

    try writer.print("HTTP/1.1 {d} {s}\r\n", .{ resp.status, resp.reason });
    try writer.print("Content-Type: {s}\r\n", .{resp.content_type});
    try writer.print("Content-Length: {d}\r\n", .{resp.body.len});
    try writer.writeAll("Connection: close\r\n");
    try writer.writeAll("\r\n");

    try stream.writeAll(fbs.getWritten());
    try stream.writeAll(resp.body);
}

fn headerValue(headers: []const Header, name_lc: []const u8) ?[]const u8 {
    for (headers) |h| {
        if (std.mem.eql(u8, h.name_lc, name_lc)) return h.value;
    }
    return null;
}

fn isWebSocketUpgrade(headers: []const Header) bool {
    const connection = headerValue(headers, "connection") orelse return false;
    const upgrade = headerValue(headers, "upgrade") orelse return false;
    if (!containsTokenCI(connection, "upgrade")) return false;
    return std.ascii.eqlIgnoreCase(upgrade, "websocket");
}

fn containsTokenCI(hdr_value: []const u8, token: []const u8) bool {
    // Split by ',' and trim.
    var it = std.mem.splitScalar(u8, hdr_value, ',');
    while (it.next()) |part| {
        const t = std.mem.trim(u8, part, " \t");
        if (std.ascii.eqlIgnoreCase(t, token)) return true;
    }
    return false;
}

fn readHttpRequest(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    scratch: *std.ArrayList(u8),
    max_header_bytes: usize,
    max_body_bytes: usize,
) !HttpRequest {
    scratch.clearRetainingCapacity();

    // Read until end of headers (\r\n\r\n)
    while (true) {
        if (scratch.items.len > max_header_bytes) return error.HeaderTooLarge;

        var tmp: [2048]u8 = undefined;
        const n = try std.posix.recv(stream.handle, &tmp, 0);
        if (n == 0) return error.UnexpectedEof;
        try scratch.appendSlice(allocator, tmp[0..n]);

        if (std.mem.indexOf(u8, scratch.items, "\r\n\r\n")) |_| break;
    }

    const header_end = std.mem.indexOf(u8, scratch.items, "\r\n\r\n").? + 4;
    const header_bytes = scratch.items[0..header_end];
    const body_start = header_end;

    // Parse request line + headers.
    var lines_it = std.mem.splitSequence(u8, header_bytes, "\r\n");
    const req_line = lines_it.next() orelse return error.BadRequest;

    var parts_it = std.mem.splitScalar(u8, req_line, ' ');
    const method = parts_it.next() orelse return error.BadRequest;
    const path = parts_it.next() orelse return error.BadRequest;

    // Collect headers.
    var headers_list: std.ArrayList(Header) = .empty;
    defer headers_list.deinit(allocator);
    errdefer {
        for (headers_list.items) |h| allocator.free(h.name_lc);
    }

    while (lines_it.next()) |line| {
        if (line.len == 0) break;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = std.mem.trim(u8, line[0..colon], " \t");
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");

        const name_lc = try allocator.alloc(u8, name.len);
        errdefer allocator.free(name_lc);
        for (name, 0..) |cch, i| name_lc[i] = std.ascii.toLower(cch);

        try headers_list.append(allocator, .{ .name_lc = name_lc, .value = value });
    }

    // Read body if present.
    const content_length_str = headerValue(headers_list.items, "content-length");
    var body_len: usize = 0;
    if (content_length_str) |s| {
        body_len = std.fmt.parseInt(usize, s, 10) catch return error.BadRequest;
    } else {
        body_len = 0;
    }

    if (body_len > max_body_bytes) return error.BodyTooLarge;

    var have: usize = scratch.items.len - body_start;
    while (have < body_len) {
        var tmp2: [4096]u8 = undefined;
        const n2 = try std.posix.recv(stream.handle, &tmp2, 0);
        if (n2 == 0) return error.UnexpectedEof;
        try scratch.appendSlice(allocator, tmp2[0..n2]);
        have = scratch.items.len - body_start;
    }

    const body = scratch.items[body_start .. body_start + body_len];

    // Move header name allocations into a single owned slice.
    const headers_owned = try allocator.alloc(Header, headers_list.items.len);
    for (headers_list.items, 0..) |h, idx| headers_owned[idx] = h;

    return .{ .method = method, .path = path, .headers = headers_owned, .body = body };
}

fn freeHeaders(allocator: std.mem.Allocator, headers: []Header) void {
    for (headers) |h| allocator.free(h.name_lc);
    allocator.free(headers);
}

fn handleWebSocket(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    headers: []const Header,
    engine: *Engine,
) !void {
    var write_buf: [16 * 1024]u8 = undefined;
    var stream_writer = stream.writer(&write_buf);
    const writer = &stream_writer.interface;
    defer stream_writer.interface.flush() catch {};

    var read_buf: [16 * 1024]u8 = undefined;
    var stream_reader = stream.reader(&read_buf);
    const reader = stream_reader.interface();

    const key = headerValue(headers, "sec-websocket-key") orelse {
        try writeJsonError(allocator, stream, 400, "Bad Request", "invalid_request_error", "missing Sec-WebSocket-Key");
        return;
    };

    const accept = try computeWebSocketAccept(allocator, key);
    defer allocator.free(accept);

    try writer.writeAll("HTTP/1.1 101 Switching Protocols\r\n");
    try writer.writeAll("Upgrade: websocket\r\n");
    try writer.writeAll("Connection: Upgrade\r\n");
    try writer.print("Sec-WebSocket-Accept: {s}\r\n", .{accept});
    try writer.writeAll("\r\n");
    try stream_writer.interface.flush();

    while (!signal.shouldExit()) {
        var frame = readWsFrame(allocator, reader, 2 * 1024 * 1024) catch |err| {
            if (err != error.UnexpectedEof) {
                std.log.err("websocket read error: {any}", .{err});
            }
            break;
        };
        defer frame.deinit(allocator);

        switch (frame.opcode) {
            .close => break,
            .ping => {
                try writeWsFrame(writer, .pong, frame.payload);
                try writer.flush();
            },
            .text => {
                // Payload is JSON request.
                var parsed = openai.parseChatCompletionRequest(allocator, frame.payload) catch {
                    const err_payload = openai.ErrorResponse{ .@"error" = .{ .message = "invalid JSON", .type = "invalid_request_error" } };
                    var buf: std.ArrayList(u8) = .empty;
                    defer buf.deinit(allocator);
                    try openai.writeJson(buf.writer(allocator), err_payload);
                    try writeWsFrame(writer, .text, buf.items);
                    try writer.flush();
                    continue;
                };
                defer parsed.deinit();

                const req = parsed.value;
                const stream_resp: bool = req.stream orelse false;

                if (!stream_resp) {
                    var resp = try engine.complete(allocator, req, null, null);
                    defer resp.deinit(allocator);

                    var buf: std.ArrayList(u8) = .empty;
                    defer buf.deinit(allocator);
                    try openai.writeJson(buf.writer(allocator), resp.value);
                    try writeWsFrame(writer, .text, buf.items);
                    try writer.flush();
                } else {
                    var ws = WsSink.init(writer, allocator, engine.modelId());
                    defer ws.deinit();
                    const id = try engine.nextIdAlloc(allocator);
                    defer allocator.free(id);
                    try ws.sendRole(id, "assistant");

                    var resp = try engine.complete(allocator, req, ws.tokenSink(), id);
                    defer resp.deinit(allocator);

                    try ws.sendFinish(resp.finishReasonString());
                    try ws.done();
                }
            },
            else => {},
        }
    }
}

fn computeWebSocketAccept(allocator: std.mem.Allocator, key: []const u8) ![]u8 {
    const guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    var sha1 = std.crypto.hash.Sha1.init(.{});
    sha1.update(key);
    sha1.update(guid);
    var digest: [20]u8 = undefined;
    sha1.final(&digest);

    var out: [std.base64.standard.Encoder.calcSize(20)]u8 = undefined;
    const encoded = std.base64.standard.Encoder.encode(&out, &digest);
    return try allocator.dupe(u8, encoded);
}

fn readNoEof(reader: *std.io.Reader, buf: []u8) !void {
    reader.readSliceAll(buf) catch |err| switch (err) {
        error.EndOfStream => return error.UnexpectedEof,
        else => return err,
    };
}

fn readWsFrame(allocator: std.mem.Allocator, reader: *std.io.Reader, max_payload: usize) !WsFrame {
    var hdr: [2]u8 = undefined;
    try readNoEof(reader, &hdr);

    const fin = (hdr[0] & 0x80) != 0;
    const opcode: WsOpcode = @enumFromInt(@as(u4, @intCast(hdr[0] & 0x0F)));
    const masked = (hdr[1] & 0x80) != 0;
    var len: u64 = hdr[1] & 0x7F;

    if (len == 126) {
        var ext: [2]u8 = undefined;
        try readNoEof(reader, &ext);
        len = (@as(u16, ext[0]) << 8) | ext[1];
    } else if (len == 127) {
        var ext8: [8]u8 = undefined;
        try readNoEof(reader, &ext8);
        len = 0;
        for (ext8) |b| len = (len << 8) | b;
    }

    if (len > max_payload) return error.PayloadTooLarge;

    var mask_key: [4]u8 = .{ 0, 0, 0, 0 };
    if (masked) try readNoEof(reader, &mask_key);

    const payload = try allocator.alloc(u8, @intCast(len));
    errdefer allocator.free(payload);
    if (payload.len > 0) try readNoEof(reader, payload);

    if (masked) {
        for (payload, 0..) |*b, i| {
            b.* ^= mask_key[i % 4];
        }
    }

    return .{ .opcode = opcode, .payload = payload, .fin = fin };
}

fn writeWsFrame(writer: anytype, opcode: WsOpcode, payload: []const u8) !void {
    var header: [14]u8 = undefined;
    header[0] = 0x80 | @as(u8, @intFromEnum(opcode));

    var header_len: usize = 2;
    if (payload.len <= 125) {
        header[1] = @as(u8, @intCast(payload.len));
    } else if (payload.len <= 0xFFFF) {
        header[1] = 126;
        header[2] = @as(u8, @intCast((payload.len >> 8) & 0xFF));
        header[3] = @as(u8, @intCast(payload.len & 0xFF));
        header_len = 4;
    } else {
        header[1] = 127;
        const len64: u64 = payload.len;
        var i: usize = 0;
        while (i < 8) : (i += 1) {
            const shift: u6 = @intCast((7 - i) * 8);
            header[2 + i] = @as(u8, @intCast((len64 >> shift) & 0xFF));
        }
        header_len = 10;
    }

    try writer.writeAll(header[0..header_len]);
    if (payload.len > 0) try writer.writeAll(payload);
}

const WsSink = struct {
    writer: *std.Io.Writer,
    allocator: std.mem.Allocator,
    id: []const u8,
    created: i64,
    model: []const u8,

    id_owned: ?[]u8 = null,

    pub fn init(writer: *std.Io.Writer, allocator: std.mem.Allocator, model: []const u8) WsSink {
        return .{ .writer = writer, .allocator = allocator, .id = "", .created = std.time.timestamp(), .model = model, .id_owned = null };
    }

    pub fn deinit(self: *WsSink) void {
        if (self.id_owned) |owned| self.allocator.free(owned);
        self.id_owned = null;
    }

    pub fn tokenSink(self: *WsSink) inference.TokenSink {
        return .{ .ctx = self, .writeFn = wsWriteToken, .flushFn = wsFlush };
    }

    pub fn sendRole(self: *WsSink, id: []const u8, role: []const u8) !void {
        if (self.id_owned) |owned| self.allocator.free(owned);
        self.id_owned = try self.allocator.dupe(u8, id);
        self.id = self.id_owned.?;
        self.created = std.time.timestamp();
        const chunk = openai.ChatCompletionChunk{
            .id = self.id,
            .object = "chat.completion.chunk",
            .created = self.created,
            .model = self.model,
            .choices = &[_]openai.ChatCompletionChunkChoice{.{
                .index = 0,
                .delta = .{ .role = role, .content = null },
                .finish_reason = null,
            }},
        };
        try self.sendChunk(chunk);
    }

    pub fn sendFinish(self: *WsSink, finish_reason: []const u8) !void {
        const chunk = openai.ChatCompletionChunk{
            .id = self.id,
            .object = "chat.completion.chunk",
            .created = self.created,
            .model = self.model,
            .choices = &[_]openai.ChatCompletionChunkChoice{.{
                .index = 0,
                .delta = .{ .role = null, .content = null },
                .finish_reason = finish_reason,
            }},
        };
        try self.sendChunk(chunk);
    }

    pub fn done(self: *WsSink) !void {
        try writeWsFrame(self.writer, .text, "[DONE]");
        try self.writer.flush();
    }

    fn sendChunk(self: *WsSink, chunk: openai.ChatCompletionChunk) !void {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(self.allocator);
        try openai.writeJson(buf.writer(self.allocator), chunk);
        try writeWsFrame(self.writer, .text, buf.items);
        try self.writer.flush();
    }

    fn wsFlush(ctx: *anyopaque) anyerror!void {
        const self: *WsSink = @ptrCast(@alignCast(ctx));
        try self.writer.flush();
    }

    fn wsWriteToken(ctx: *anyopaque, bytes: []const u8) anyerror!void {
        const self: *WsSink = @ptrCast(@alignCast(ctx));
        const chunk = openai.ChatCompletionChunk{
            .id = self.id,
            .object = "chat.completion.chunk",
            .created = self.created,
            .model = self.model,
            .choices = &[_]openai.ChatCompletionChunkChoice{.{
                .index = 0,
                .delta = .{ .role = null, .content = bytes },
                .finish_reason = null,
            }},
        };
        try self.sendChunk(chunk);
    }
};

/// Minimal engine for server mode.
const Engine = struct {
    model_path: []const u8,

    model: llama_cpp.Model,
    ctx: llama_cpp.Context,
    vocab: *const llama_cpp.c.llama_vocab,
    tmpl: ?[*:0]const u8,

    batch: llama_cpp.Batch,

    cfg: EngineConfig,
    grammar_z: ?[:0]u8,
    grammar_root_z: ?[:0]u8,

    id_counter: u64,

    mutex: std.Thread.Mutex = .{},
    cached_tokens: std.ArrayList(llama_cpp.Token),

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, cfg: EngineConfig) !Engine {
        const path_z = try llama_cpp.dupeZ(allocator, model_path);
        defer allocator.free(path_z);

        var mparams = llama_cpp.c.llama_model_default_params();
        mparams.n_gpu_layers = cfg.n_gpu_layers;
        mparams.use_mmap = true;
        mparams.use_mlock = false;

        const model = try llama_cpp.Model.load(path_z, mparams);
        errdefer model.deinit();

        var cparams = llama_cpp.c.llama_context_default_params();
        cparams.n_ctx = cfg.n_ctx;
        cparams.n_batch = 1024;
        cparams.n_ubatch = 512;
        cparams.n_seq_max = 1;
        cparams.offload_kqv = true;

        const cpu_count: i32 = @intCast(std.Thread.getCpuCount() catch 4);
        const final_threads = cfg.threads orelse cpu_count;
        cparams.n_threads = final_threads;
        cparams.n_threads_batch = final_threads;

        const ctx = try llama_cpp.Context.init(model, cparams);
        errdefer ctx.deinit();

        const vocab = model.vocab() orelse return error.ModelLoadFailed;
        const tmpl = model.chatTemplate();

        var grammar_z: ?[:0]u8 = null;
        var grammar_root_z: ?[:0]u8 = null;
        if (cfg.grammar_path) |gp| {
            const bytes = try std.fs.cwd().readFileAlloc(allocator, gp, 4 * 1024 * 1024);
            defer allocator.free(bytes);
            grammar_z = try chat.dupeZ(allocator, bytes);
            grammar_root_z = try chat.dupeZ(allocator, cfg.grammar_root);
        }

        return .{
            .model_path = model_path,
            .model = model,
            .ctx = ctx,
            .vocab = vocab,
            .tmpl = tmpl,
            .batch = llama_cpp.Batch.init(1024, 0, 1),
            .cfg = cfg,
            .grammar_z = grammar_z,
            .grammar_root_z = grammar_root_z,
            .id_counter = 1,
            .cached_tokens = .{},
        };
    }

    pub fn deinit(self: *Engine, allocator: std.mem.Allocator) void {
        self.cached_tokens.deinit(allocator);
        self.batch.deinit();
        self.ctx.deinit();
        self.model.deinit();
        if (self.grammar_z) |g| allocator.free(g);
        if (self.grammar_root_z) |r| allocator.free(r);
    }

    pub fn modelId(self: *Engine) []const u8 {
        return std.fs.path.basename(self.model_path);
    }

    pub fn nextIdAlloc(self: *Engine, allocator: std.mem.Allocator) ![]u8 {
        // Produces an ID like chatcmpl-000000000001 (hex).
        var buf: [32]u8 = undefined;
        const n = try std.fmt.bufPrint(&buf, "chatcmpl-{x:0>12}", .{self.id_counter});
        self.id_counter += 1;
        return try allocator.dupe(u8, n);
    }

    pub const Completion = struct {
        value: openai.ChatCompletionResponse,
        finish_reason: []const u8,

        pub fn deinit(self: *Completion, allocator: std.mem.Allocator) void {
            allocator.free(self.value.id);
            allocator.free(self.value.model);
            if (self.value.choices.len > 0) {
                allocator.free(self.value.choices[0].message.content);
            }
            allocator.free(self.value.choices);
        }

        pub fn finishReasonString(self: *Completion) []const u8 {
            return self.finish_reason;
        }
    };

    pub fn complete(
        self: *Engine,
        allocator: std.mem.Allocator,
        req: openai.ChatCompletionRequest,
        sink: ?inference.TokenSink,
        forced_id: ?[]const u8,
    ) !Completion {
        // Convert request messages to internal representation.
        var msgs: std.ArrayList(chat.Message) = .empty;
        defer chat.deinitMessages(allocator, &msgs);

        for (req.messages) |m| {
            const role = if (std.ascii.eqlIgnoreCase(m.role, "system")) chat.Role.system else if (std.ascii.eqlIgnoreCase(m.role, "user")) chat.Role.user else if (std.ascii.eqlIgnoreCase(m.role, "assistant")) chat.Role.assistant else return error.InvalidRole;
            const content_z = try chat.dupeZ(allocator, m.content);
            try msgs.append(allocator, .{ .role = role, .content = content_z });
        }

        // Build prompt and trim to fit context.
        const ctx_reserve: usize = 256;
        const ctx_limit: usize = @as(usize, @intCast(self.ctx.nCtx())) - ctx_reserve;

        var prompt = try inference.buildPrompt(allocator, self.tmpl, self.vocab, msgs.items);
        while (prompt.tokens.len > ctx_limit) {
            prompt.deinit(allocator);
            if (!chat.dropOldestNonSystem(&msgs, allocator)) break;
            prompt = try inference.buildPrompt(allocator, self.tmpl, self.vocab, msgs.items);
        }
        defer prompt.deinit(allocator);

        if (prompt.tokens.len > ctx_limit) {
            return error.ContextTooSmall;
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        // Calculate common prefix with cached tokens.
        var n_past: usize = 0;
        const n_common = @min(self.cached_tokens.items.len, prompt.tokens.len);
        for (0..n_common) |i| {
            if (self.cached_tokens.items[i] != prompt.tokens[i]) break;
            n_past += 1;
        }

        // Reset KV cache after the common prefix.
        if (n_past < self.cached_tokens.items.len) {
            self.ctx.kvCacheSeqRm(0, @intCast(n_past), -1);
            self.cached_tokens.shrinkRetainingCapacity(n_past);
        }

        // Sampler config with request overrides.
        const s_cfg = llama_cpp.SamplerConfig{
            .temp = req.temperature orelse self.cfg.temp,
            .top_k = self.cfg.top_k,
            .top_p = req.top_p orelse self.cfg.top_p,
            .min_p = self.cfg.min_p,
            .seed = req.seed orelse self.cfg.seed,
        };

        var sampler: llama_cpp.Sampler = undefined;
        if (self.grammar_z) |g| {
            const default_root: [:0]const u8 = "root";
            const root = self.grammar_root_z orelse default_root;
            sampler = try llama_cpp.Sampler.initWithConfigAndGrammar(s_cfg, self.vocab, g, root);
        } else {
            sampler = try llama_cpp.Sampler.initWithConfig(s_cfg);
        }
        defer sampler.deinit();
        sampler.reset();

        const max_tokens: usize = @intCast(req.max_tokens orelse 4096);

        const gen = try inference.generate(allocator, self.ctx, self.vocab, &self.batch, sampler, prompt.tokens, .{
            .max_tokens = max_tokens,
            .sink = sink,
            .shouldStopCtx = null,
            .shouldStopFn = null,
            .n_past = n_past,
        });
        defer gen.deinit(allocator);

        // Update cache with new tokens.
        // The prompt tokens after n_past are new, plus the generated tokens.
        if (n_past < prompt.tokens.len) {
            try self.cached_tokens.appendSlice(allocator, prompt.tokens[n_past..]);
        }
        // Generated text needs to be tokenized to append to cache, but we don't have the token IDs here easily
        // because generate() returns text.
        // Wait, generate() loop has the token IDs. Ideally generate() should return the generated token IDs.
        // For now, we only cache the prompt part for the NEXT turn.
        // If we want to cache the assistant response too, we need the token IDs.
        // Let's rely on re-tokenization in the next request which will match the prompt + assistant response.
        // Actually, if we don't cache the assistant output, the next request (which includes it) will see a mismatch
        // after the user prompt, but that's okay. It will still save the system prompt + user prompt prefix.
        // A better optimization would be to have generate() return tokens or append to cache.
        // For now, let's stick to prompt caching which is the biggest win (system prompt).

        const finish_reason = switch (gen.finish_reason) {
            .stop => "stop",
            .length => "length",
            .context_limit => "length",
            .aborted => "stop",
        };

        const id = if (forced_id) |fid| try allocator.dupe(u8, fid) else try self.nextIdAlloc(allocator);
        const model_name = try allocator.dupe(u8, req.model orelse self.modelId());
        const msg_content = try allocator.dupe(u8, gen.text);

        const choice = openai.ChatCompletionChoice{
            .index = 0,
            .message = .{ .role = "assistant", .content = msg_content },
            .finish_reason = finish_reason,
        };

        const choices = try allocator.alloc(openai.ChatCompletionChoice, 1);
        choices[0] = choice;

        const usage = openai.Usage{
            .prompt_tokens = gen.prompt_tokens,
            .completion_tokens = gen.completion_tokens,
            .total_tokens = gen.prompt_tokens + gen.completion_tokens,
        };

        const resp = openai.ChatCompletionResponse{
            .id = id,
            .object = "chat.completion",
            .created = std.time.timestamp(),
            .model = model_name,
            .choices = choices,
            .usage = usage,
        };

        return .{ .value = resp, .finish_reason = finish_reason };
    }
};
