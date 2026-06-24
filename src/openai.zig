const std = @import("std");

/// OpenAI-compatible types for `/v1/chat/completions`.
///
/// We intentionally ignore unknown fields so that common OpenAI client SDKs work
/// without needing exact schema parity.
pub const Role = enum { system, user, assistant, tool };

pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
    name: ?[]const u8 = null,
};

pub const ChatCompletionRequest = struct {
    model: ?[]const u8 = null,
    messages: []ChatMessage,

    temperature: ?f32 = null,
    top_p: ?f32 = null,
    max_tokens: ?u32 = null,
    stream: ?bool = null,
    seed: ?u32 = null,
};

pub const Usage = struct {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
};

pub const ChatCompletionChoice = struct {
    index: usize,
    message: struct {
        role: []const u8,
        content: []const u8,
    },
    finish_reason: []const u8,
};

pub const ChatCompletionResponse = struct {
    id: []const u8,
    object: []const u8,
    created: i64,
    model: []const u8,
    choices: []const ChatCompletionChoice,
    usage: Usage,
};

pub const ChatCompletionChunkChoice = struct {
    index: usize,
    delta: struct {
        role: ?[]const u8 = null,
        content: ?[]const u8 = null,
    },
    finish_reason: ?[]const u8 = null,
};

pub const ChatCompletionChunk = struct {
    id: []const u8,
    object: []const u8,
    created: i64,
    model: []const u8,
    choices: []const ChatCompletionChunkChoice,
};

/// Legacy `/v1/completions` (text completion). `prompt` accepted as a string
/// only (array prompts not supported).
pub const CompletionRequest = struct {
    model: ?[]const u8 = null,
    prompt: []const u8,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    max_tokens: ?u32 = null,
    stream: ?bool = null,
    seed: ?u32 = null,
};

pub const CompletionChoice = struct {
    text: []const u8,
    index: usize,
    finish_reason: []const u8,
};

pub const CompletionResponse = struct {
    id: []const u8,
    object: []const u8,
    created: i64,
    model: []const u8,
    choices: []const CompletionChoice,
    usage: Usage,
};

pub const ErrorResponse = struct {
    @"error": struct {
        message: []const u8,
        type: []const u8,
        param: ?[]const u8 = null,
        code: ?[]const u8 = null,
    },
};

pub const ParseError = error{
    InvalidJson,
    MissingMessages,
    InvalidTemperature,
    InvalidTopP,
    InvalidMaxTokens,
    InvalidRole,
    EmptyContent,
};

/// Field-level descriptor for a validation failure. Lets the HTTP layer turn
/// any `ParseError` into an OpenAI-style 400 with `param` populated.
pub const FieldError = struct {
    err: ParseError,
    param: []const u8,
    message: []const u8,
};

/// Map a `ParseError` produced by `parseChatCompletionRequest` to a stable
/// (message, param) pair suitable for an `invalid_request_error` JSON body.
pub fn describeParseError(err: ParseError) FieldError {
    return switch (err) {
        ParseError.InvalidJson => .{ .err = err, .param = "body", .message = "invalid JSON" },
        ParseError.MissingMessages => .{ .err = err, .param = "messages", .message = "messages must be a non-empty array" },
        ParseError.InvalidTemperature => .{ .err = err, .param = "temperature", .message = "temperature must be in [0, 2]" },
        ParseError.InvalidTopP => .{ .err = err, .param = "top_p", .message = "top_p must be in (0, 1]" },
        ParseError.InvalidMaxTokens => .{ .err = err, .param = "max_tokens", .message = "max_tokens must be > 0 and <= 1048576" },
        ParseError.InvalidRole => .{ .err = err, .param = "messages.role", .message = "role must be one of: system, user, assistant, tool" },
        ParseError.EmptyContent => .{ .err = err, .param = "messages.content", .message = "message content must not be empty" },
    };
}

/// Parse a JSON request body into `ChatCompletionRequest`.
///
/// - Requires `messages` to be present.
/// - Unknown fields are ignored.
/// - Validates parameter ranges (temperature, top_p, max_tokens) and per-message
///   role/content so callers receive a clean 400 instead of an opaque server
///   error from the sampler or tokenizer downstream.
pub fn parseChatCompletionRequest(
    allocator: std.mem.Allocator,
    body: []const u8,
) !std.json.Parsed(ChatCompletionRequest) {
    var parsed = std.json.parseFromSlice(
        ChatCompletionRequest,
        allocator,
        body,
        .{ .ignore_unknown_fields = true },
    ) catch return ParseError.InvalidJson;
    errdefer parsed.deinit();

    if (parsed.value.messages.len == 0) return ParseError.MissingMessages;

    if (parsed.value.temperature) |t| {
        if (!std.math.isFinite(t) or t < 0.0 or t > 2.0) return ParseError.InvalidTemperature;
    }
    if (parsed.value.top_p) |p| {
        if (!std.math.isFinite(p) or p <= 0.0 or p > 1.0) return ParseError.InvalidTopP;
    }
    if (parsed.value.max_tokens) |m| {
        if (m == 0 or m > 1024 * 1024) return ParseError.InvalidMaxTokens;
    }

    for (parsed.value.messages) |m| {
        if (m.content.len == 0) return ParseError.EmptyContent;
        const role_ok = std.ascii.eqlIgnoreCase(m.role, "system") or
            std.ascii.eqlIgnoreCase(m.role, "user") or
            std.ascii.eqlIgnoreCase(m.role, "assistant");
        if (!role_ok) return ParseError.InvalidRole;
    }

    return parsed;
}

/// Parse a `/v1/completions` request body. `prompt` is required and must be a
/// non-empty string; ranges validated like the chat endpoint.
pub fn parseCompletionRequest(
    allocator: std.mem.Allocator,
    body: []const u8,
) !std.json.Parsed(CompletionRequest) {
    var parsed = std.json.parseFromSlice(
        CompletionRequest,
        allocator,
        body,
        .{ .ignore_unknown_fields = true },
    ) catch return ParseError.InvalidJson;
    errdefer parsed.deinit();

    if (parsed.value.prompt.len == 0) return ParseError.EmptyContent;
    if (parsed.value.temperature) |t| {
        if (!std.math.isFinite(t) or t < 0.0 or t > 2.0) return ParseError.InvalidTemperature;
    }
    if (parsed.value.top_p) |p| {
        if (!std.math.isFinite(p) or p <= 0.0 or p > 1.0) return ParseError.InvalidTopP;
    }
    if (parsed.value.max_tokens) |m| {
        if (m == 0 or m > 1024 * 1024) return ParseError.InvalidMaxTokens;
    }
    return parsed;
}

/// Write JSON with stable settings.
pub fn writeJson(writer: anytype, value: anytype) !void {
    const WriterIface = std.io.Writer;
    const WT = @TypeOf(writer);

    if (WT == *WriterIface) {
        var jw = std.json.Stringify{ .writer = writer, .options = .{ .whitespace = .minified } };
        try jw.write(value);
        return;
    }

    // Bridge deprecated/legacy writers (e.g. std.io.GenericWriter) to the new Writer API.
    var w = writer;
    if (@hasDecl(@TypeOf(w), "adaptToNewApi")) {
        var buf: [8 * 1024]u8 = undefined;
        var adapter = w.adaptToNewApi(&buf);
        var jw = std.json.Stringify{ .writer = &adapter.new_interface, .options = .{ .whitespace = .minified } };
        try jw.write(value);
        // Flush buffered bytes from the adapter into the underlying writer.
        // Without this, JSON smaller than the adapter buffer (8 KiB) never
        // reaches the destination and the response body comes out empty.
        try adapter.new_interface.flush();
        if (adapter.err) |err| return err;
        return;
    }

    @compileError("openai.writeJson: unsupported writer type; pass *std.io.Writer or a writer supporting adaptToNewApi()");
}

// ---------------------------------------------------------------------------
// Tests for request validation. These guard the OpenAI HTTP boundary so bad
// client input never reaches the sampler/engine.
// ---------------------------------------------------------------------------

test "parse: rejects invalid JSON" {
    const t = std.testing;
    try t.expectError(ParseError.InvalidJson, parseChatCompletionRequest(t.allocator, "{not json"));
}

test "parse: rejects empty messages" {
    const t = std.testing;
    const body =
        \\{"messages":[]}
    ;
    try t.expectError(ParseError.MissingMessages, parseChatCompletionRequest(t.allocator, body));
}

test "parse: rejects out-of-range temperature" {
    const t = std.testing;
    const body =
        \\{"messages":[{"role":"user","content":"hi"}],"temperature":3.5}
    ;
    try t.expectError(ParseError.InvalidTemperature, parseChatCompletionRequest(t.allocator, body));
}

test "parse: rejects out-of-range top_p" {
    const t = std.testing;
    const body =
        \\{"messages":[{"role":"user","content":"hi"}],"top_p":0}
    ;
    try t.expectError(ParseError.InvalidTopP, parseChatCompletionRequest(t.allocator, body));
}

test "parse: rejects zero max_tokens" {
    const t = std.testing;
    const body =
        \\{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}
    ;
    try t.expectError(ParseError.InvalidMaxTokens, parseChatCompletionRequest(t.allocator, body));
}

test "parse: rejects unknown role" {
    const t = std.testing;
    const body =
        \\{"messages":[{"role":"alien","content":"hi"}]}
    ;
    try t.expectError(ParseError.InvalidRole, parseChatCompletionRequest(t.allocator, body));
}

test "parse: rejects empty content" {
    const t = std.testing;
    const body =
        \\{"messages":[{"role":"user","content":""}]}
    ;
    try t.expectError(ParseError.EmptyContent, parseChatCompletionRequest(t.allocator, body));
}

test "parse: accepts well-formed request" {
    const t = std.testing;
    const body =
        \\{"messages":[{"role":"user","content":"hi"}],"temperature":0.7,"top_p":0.9,"max_tokens":128}
    ;
    var parsed = try parseChatCompletionRequest(t.allocator, body);
    defer parsed.deinit();
    try t.expectEqual(@as(usize, 1), parsed.value.messages.len);
    try t.expectEqualStrings("user", parsed.value.messages[0].role);
}

test "parse completions: accepts prompt, rejects empty" {
    const t = std.testing;
    var parsed = try parseCompletionRequest(t.allocator, "{\"prompt\":\"hello\",\"max_tokens\":16}");
    defer parsed.deinit();
    try t.expectEqualStrings("hello", parsed.value.prompt);
    try t.expectError(ParseError.EmptyContent, parseCompletionRequest(t.allocator, "{\"prompt\":\"\"}"));
    try t.expectError(ParseError.InvalidJson, parseCompletionRequest(t.allocator, "{\"model\":\"x\"}"));
}

test "describeParseError: every variant has a non-empty message and param" {
    const variants = [_]ParseError{
        ParseError.InvalidJson,
        ParseError.MissingMessages,
        ParseError.InvalidTemperature,
        ParseError.InvalidTopP,
        ParseError.InvalidMaxTokens,
        ParseError.InvalidRole,
        ParseError.EmptyContent,
    };
    for (variants) |v| {
        const fe = describeParseError(v);
        try std.testing.expect(fe.message.len > 0);
        try std.testing.expect(fe.param.len > 0);
    }
}
