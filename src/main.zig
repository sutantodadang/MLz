const std = @import("std");
const llama_cpp = @import("llama_cpp.zig");
const chat = @import("chat.zig");
const signal = @import("signal.zig");
const terminal = @import("terminal.zig");
const server = @import("server.zig");
const config = @import("config.zig");
const inference = @import("inference.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Parse configuration
    const cfg = config.Config.parse(allocator, args) catch |err| {
        switch (err) {
            error.MissingModelPath => {
                std.debug.print(
                    \\Usage: {s} <model_path> [options]
                    \\
                    \\Positional:
                    \\  <model_path>       Path to GGUF model file
                    \\
                    \\Options:
                    \\  --prompt <string>     Single prompt mode (non-interactive)
                    \\  --temp <float>        Temperature (default: 0.8)
                    \\  --top-k <int>         Top-K (default: 40)
                    \\  --top-p <float>       Top-P (default: 0.95)
                    \\  --min-p <float>       Min-P (default: 0.05)
                    \\  --seed <int>          Seed (default: 42)
                    \\  --ctx <int>           Context size (default: 4096)
                    \\  --ngl <int>           GPU layers (default: 999)
                    \\  --threads <int>       Number of threads (default: CPU count)
                    \\  --stream <bool>       Stream output (default: true)
                    \\  --system <string>     System prompt
                    \\  --load-chat <file>    Load conversation from JSON
                    \\  --save-chat <file>    Save conversation to JSON
                    \\  --grammar <file>      Constrain output with a GBNF grammar file
                    \\  --grammar-root <name> Start rule name for grammar (default: root)
                    \\  --server              Run as an OpenAI-compatible server
                    \\  --host <string>        Server host (default: 127.0.0.1)
                    \\  --port <int>           Server port (default: 8080)
                    \\  --api-key <string>     Require Authorization: Bearer <api-key>
                    \\
                , .{args[0]});
                return;
            },
            else => return err,
        }
    };

    const model_path = cfg.model_path;

    if (cfg.server_mode) {
        try server.run(allocator, model_path, .{
            .host = cfg.server_host,
            .port = cfg.server_port,
            .api_key = cfg.server_api_key,
        }, .{
            .n_ctx = cfg.n_ctx,
            .n_gpu_layers = cfg.n_gpu_layers,
            .threads = cfg.threads,
            .temp = cfg.temp,
            .top_k = cfg.top_k,
            .top_p = cfg.top_p,
            .min_p = cfg.min_p,
            .seed = cfg.seed,
            .grammar_path = cfg.grammar_path,
            .grammar_root = cfg.grammar_root,
            .draft_model_path = cfg.draft_model_path,
        });
        return;
    }

    // Use the generic Engine for both interactive and one-shot modes.
    const engine_cfg = server.EngineConfig{
        .n_ctx = cfg.n_ctx,
        .n_gpu_layers = cfg.n_gpu_layers,
        .threads = cfg.threads,
        .temp = cfg.temp,
        .top_k = cfg.top_k,
        .top_p = cfg.top_p,
        .min_p = cfg.min_p,
        .seed = cfg.seed,
        .grammar_path = cfg.grammar_path,
        .grammar_root = cfg.grammar_root,
        .draft_model_path = cfg.draft_model_path,
    };

    var engine = try server.Engine.init(allocator, model_path, engine_cfg);
    defer engine.deinit(allocator);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    const stdin_reader = std.fs.File.stdin().deprecatedReader();
    const stdin_is_tty = std.fs.File.stdin().isTty();
    const use_color = terminal.enableAnsiColors();

    // Prepare message history
    var msgs: std.ArrayList(chat.Message) = .empty;
    defer chat.deinitMessages(allocator, &msgs);

    if (cfg.load_chat_path) |load_path| {
        msgs = chat.loadJson(allocator, load_path) catch |err| switch (err) {
            error.FileNotFound => .empty,
            else => return err,
        };
    }
    // Note: We don't save immediately on load, only after turns.

    if (cfg.system_prompt) |sys| {
        try chat.setOrPrependSystemPrompt(allocator, &msgs, sys);
    }

    // One-shot Prompt Mode
    if (cfg.prompt_mode) {
        if (cfg.user_prompt) |input| {
            const user_z = try chat.dupeZ(allocator, input);
            errdefer allocator.free(user_z);
            try msgs.append(allocator, .{ .role = .user, .content = user_z });

            var dummy_ctx: u8 = 0;
            const sink: ?inference.TokenSink = if (cfg.stream) .{ .ctx = &dummy_ctx, .writeFn = printToken } else null;

            const result = try engine.chat(allocator, msgs.items, .{
                .sink = sink,
            });
            // Result owns text and tokens, usually managed by arena or manually.
            // inference.GenerationResult deinit? It has text and tokens slices.
            // However, inference.generate doesn't provide deinit for GenerationResult?
            // In engine.zig I saw defer gen.deinit(allocator).
            // Let's assume yes.
            // Wait, inference.zig defines GenerationResult?
            // Engine.chat returns GenerationResult.
            // Engine.complete calls gen.deinit(allocator). So it exists.
            defer result.deinit(allocator);

            if (!cfg.stream) {
                try stdout.print("{s}\n", .{result.text});
            } else {
                try stdout.writeAll("\n");
            }
        }
        return;
    }

    // Interactive Mode
    try stdout.print("\nMLz Llama Chat Interface\n", .{});
    try stdout.print("Model: {s}\n", .{engine.modelId()});
    try stdout.print("Context Size: {d}\n", .{engine_cfg.n_ctx});
    try stdout.print("Sampler: temp={d:.2}, top_k={d}, top_p={d:.2}, min_p={d:.2}, seed={d}\n", .{
        engine_cfg.temp, engine_cfg.top_k, engine_cfg.top_p, engine_cfg.min_p, engine_cfg.seed,
    });
    try stdout.print("Commands: /clear, /reset, exit\n\n", .{});

    var line_buf: std.ArrayList(u8) = .empty;
    defer line_buf.deinit(allocator);

    while (true) {
        if (signal.shouldExit()) break;

        try writeLabel(stdout, use_color, terminal.Ansi.green, "You");
        try stdout.writeAll(": ");

        line_buf.clearRetainingCapacity();
        stdin_reader.streamUntilDelimiter(line_buf.writer(allocator), '\n', null) catch |err| {
            if (err == error.EndOfStream) break;
            if (signal.shouldExit()) break;
            return err;
        };
        const user_input = std.mem.trim(u8, line_buf.items, " \r\n");

        if (!stdin_is_tty) try stdout.writeAll("\n");
        if (user_input.len == 0) continue;
        if (std.mem.eql(u8, user_input, "exit")) break;

        if (std.mem.eql(u8, user_input, "/clear") or std.mem.eql(u8, user_input, "/reset")) {
            chat.clearKeepSystem(allocator, &msgs);
            engine.reset();
            maybeSaveChat(allocator, cfg.save_chat_path, msgs.items);
            try writeDim(stdout, use_color);
            try stdout.writeAll("[conversation cleared]\n\n");
            try writeReset(stdout, use_color);
            continue;
        }

        // User turn
        const user_z = try chat.dupeZ(allocator, user_input);
        errdefer allocator.free(user_z);
        try msgs.append(allocator, .{ .role = .user, .content = user_z });

        // Assistant turn
        try writeLabel(stdout, use_color, terminal.Ansi.cyan, "Assistant");
        try stdout.writeAll(": ");

        var dummy_ctx: u8 = 0;
        const sink: ?inference.TokenSink = if (cfg.stream) .{ .ctx = &dummy_ctx, .writeFn = printToken } else null;
        var gen_timer = try std.time.Timer.start();

        const result = engine.chat(allocator, msgs.items, .{
            .sink = sink,
        }) catch |err| {
            if (err == error.ContextTooSmall) {
                try stdout.print("\n[Error: Input too large for context]\n", .{});
                // Remove the user message we just added
                if (msgs.pop()) |removed| {
                    allocator.free(removed.content);
                }
                continue;
            }
            return err;
        };
        defer result.deinit(allocator);

        if (!cfg.stream) {
            try stdout.print("{s}", .{result.text});
        }
        try stdout.writeAll("\n\n");

        const gen_total_ns = gen_timer.read();

        // Calculate stats
        const prompt_tokens = result.prompt_tokens;
        const completion_tokens = result.completion_tokens;
        const ctx_used = prompt_tokens + completion_tokens;
        const ctx_remaining = if (engine_cfg.n_ctx > ctx_used) engine_cfg.n_ctx - ctx_used else 0;

        try printTurnStats(
            stdout,
            use_color,
            prompt_tokens,
            completion_tokens,
            ctx_remaining,
            null, // ttft lost
            gen_total_ns,
        );

        // Append assistant response to history
        const asst_z = try chat.dupeZ(allocator, result.text);
        errdefer allocator.free(asst_z);
        try msgs.append(allocator, .{ .role = .assistant, .content = asst_z });

        maybeSaveChat(allocator, cfg.save_chat_path, msgs.items);
    }
}

fn printToken(ctx: *anyopaque, bytes: []const u8) anyerror!void {
    _ = ctx;
    try std.fs.File.stdout().deprecatedWriter().writeAll(bytes);
}

fn maybeSaveChat(
    allocator: std.mem.Allocator,
    save_path: ?[]const u8,
    msgs: []const chat.Message,
) void {
    if (save_path) |p| {
        chat.saveJson(allocator, p, msgs) catch |err| {
            std.log.err("failed to save chat to '{s}': {any}", .{ p, err });
        };
    }
}

fn writeLabel(writer: anytype, enabled: bool, color: []const u8, label: []const u8) !void {
    if (enabled) try writer.writeAll(color);
    try writer.writeAll(label);
    if (enabled) try writer.writeAll(terminal.Ansi.reset);
}

fn writeDim(writer: anytype, enabled: bool) !void {
    if (enabled) try writer.writeAll(terminal.Ansi.dim);
}

fn writeReset(writer: anytype, enabled: bool) !void {
    if (enabled) try writer.writeAll(terminal.Ansi.reset);
}

/// Prints per-turn token counts and inference metrics.
fn printTurnStats(
    writer: anytype,
    enabled_color: bool,
    prompt_tokens: usize,
    completion_tokens: usize,
    ctx_remaining: usize,
    ttft_ns: ?u64,
    total_ns: u64,
) !void {
    const total_s: f64 = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
    const ttft_ms: ?f64 = if (ttft_ns) |ns| @as(f64, @floatFromInt(ns)) / 1_000_000.0 else null;
    const tok_s: f64 = if (total_s > 0) @as(f64, @floatFromInt(completion_tokens)) / total_s else 0.0;

    try writeDim(writer, enabled_color);
    if (ttft_ms) |ms| {
        try writer.print(
            "[tokens prompt={d} completion={d} ctx_remaining={d}] [perf ttft={d:.1}ms total={d:.2}s tok/s={d:.2}]\n\n",
            .{ prompt_tokens, completion_tokens, ctx_remaining, ms, total_s, tok_s },
        );
    } else {
        try writer.print(
            "[tokens prompt={d} completion={d} ctx_remaining={d}] [perf ttft=n/a total={d:.2}s tok/s={d:.2}]\n\n",
            .{ prompt_tokens, completion_tokens, ctx_remaining, total_s, tok_s },
        );
    }
    try writeReset(writer, enabled_color);
}
