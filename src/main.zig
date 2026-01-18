const std = @import("std");
const llama_cpp = @import("llama_cpp.zig");
const chat = @import("chat.zig");
const signal = @import("signal.zig");
const terminal = @import("terminal.zig");
const server = @import("server.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var path: ?[]const u8 = null;

    // Default configuration
    var temp: f32 = 0.8;
    var top_k: i32 = 40;
    var top_p: f32 = 0.95;
    var min_p: f32 = 0.05;
    var seed: u32 = 42;
    var n_ctx: u32 = 4096;
    var n_gpu_layers: i32 = 999;
    var threads: ?i32 = null;
    var stream: bool = true;
    var system_prompt: ?[]const u8 = null;
    var save_chat_path: ?[]const u8 = null;
    var load_chat_path: ?[]const u8 = null;
    var grammar_path: ?[]const u8 = null;
    var grammar_root: []const u8 = "root";

    // Server mode configuration
    var server_mode: bool = false;
    var server_host: []const u8 = "127.0.0.1";
    var server_port: u16 = 8080;
    var server_api_key: ?[]const u8 = null;

    var prompt_mode: bool = false;
    var user_prompt: ?[]const u8 = null;

    var arg_idx: usize = 1;
    while (arg_idx < args.len) : (arg_idx += 1) {
        const arg = args[arg_idx];
        if (std.mem.eql(u8, arg, "--temp")) {
            arg_idx += 1;
            if (arg_idx < args.len) temp = std.fmt.parseFloat(f32, args[arg_idx]) catch temp;
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            arg_idx += 1;
            if (arg_idx < args.len) {
                user_prompt = args[arg_idx];
                prompt_mode = true;
            }
            arg_idx += 1;
            if (arg_idx < args.len) top_k = std.fmt.parseInt(i32, args[arg_idx], 10) catch top_k;
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            arg_idx += 1;
            if (arg_idx < args.len) top_p = std.fmt.parseFloat(f32, args[arg_idx]) catch top_p;
        } else if (std.mem.eql(u8, arg, "--min-p")) {
            arg_idx += 1;
            if (arg_idx < args.len) min_p = std.fmt.parseFloat(f32, args[arg_idx]) catch min_p;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            arg_idx += 1;
            if (arg_idx < args.len) seed = std.fmt.parseInt(u32, args[arg_idx], 10) catch seed;
        } else if (std.mem.eql(u8, arg, "--ctx")) {
            arg_idx += 1;
            if (arg_idx < args.len) n_ctx = std.fmt.parseInt(u32, args[arg_idx], 10) catch n_ctx;
        } else if (std.mem.eql(u8, arg, "--ngl")) {
            arg_idx += 1;
            if (arg_idx < args.len) n_gpu_layers = std.fmt.parseInt(i32, args[arg_idx], 10) catch n_gpu_layers;
        } else if (std.mem.eql(u8, arg, "--threads")) {
            arg_idx += 1;
            if (arg_idx < args.len) threads = std.fmt.parseInt(i32, args[arg_idx], 10) catch threads;
        } else if (std.mem.eql(u8, arg, "--stream")) {
            arg_idx += 1;
            if (arg_idx < args.len) {
                const val = args[arg_idx];
                stream = std.mem.eql(u8, val, "true") or std.mem.eql(u8, val, "1");
            }
        } else if (std.mem.eql(u8, arg, "--system")) {
            arg_idx += 1;
            if (arg_idx < args.len) system_prompt = args[arg_idx];
        } else if (std.mem.eql(u8, arg, "--save-chat")) {
            arg_idx += 1;
            if (arg_idx < args.len) save_chat_path = args[arg_idx];
        } else if (std.mem.eql(u8, arg, "--load-chat")) {
            arg_idx += 1;
            if (arg_idx < args.len) load_chat_path = args[arg_idx];
        } else if (std.mem.eql(u8, arg, "--grammar")) {
            arg_idx += 1;
            if (arg_idx < args.len) grammar_path = args[arg_idx];
        } else if (std.mem.eql(u8, arg, "--grammar-root")) {
            arg_idx += 1;
            if (arg_idx < args.len) grammar_root = args[arg_idx];
        } else if (std.mem.eql(u8, arg, "--server")) {
            server_mode = true;
        } else if (std.mem.eql(u8, arg, "--host")) {
            arg_idx += 1;
            if (arg_idx < args.len) server_host = args[arg_idx];
        } else if (std.mem.eql(u8, arg, "--port")) {
            arg_idx += 1;
            if (arg_idx < args.len) server_port = std.fmt.parseInt(u16, args[arg_idx], 10) catch server_port;
        } else if (std.mem.eql(u8, arg, "--api-key")) {
            arg_idx += 1;
            if (arg_idx < args.len) server_api_key = args[arg_idx];
        } else if (!std.mem.startsWith(u8, arg, "--")) {
            if (path == null) {
                path = arg;
            }
        }
    }

    if (path == null) {
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
    }

    const model_path = path.?;

    if (server_mode) {
        try server.run(allocator, model_path, .{
            .host = server_host,
            .port = server_port,
            .api_key = server_api_key,
        }, .{
            .n_ctx = n_ctx,
            .n_gpu_layers = n_gpu_layers,
            .threads = threads,
            .temp = temp,
            .top_k = top_k,
            .top_p = top_p,
            .min_p = min_p,
            .seed = seed,
            .grammar_path = grammar_path,
            .grammar_root = grammar_root,
        });
        return;
    }

    std.debug.print("Initializing llama.cpp backend...\n", .{});
    const backend = llama_cpp.Backend.init();
    defer backend.deinit();

    const ctrlc = try signal.CtrlC.init();
    defer ctrlc.deinit();

    const sys_info = try llama_cpp.systemInfo(allocator);
    defer allocator.free(sys_info);
    std.debug.print("System Info: {s}\n", .{sys_info});

    const path_z = try llama_cpp.dupeZ(allocator, model_path);
    defer allocator.free(path_z);

    // Optimized model parameters
    var mparams = llama_cpp.c.llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.use_mmap = true;
    mparams.use_mlock = false;

    std.debug.print("Loading model: {s}...\n", .{model_path});
    const model = try llama_cpp.Model.load(path_z, mparams);
    defer model.deinit();

    // Optimized context parameters
    var cparams = llama_cpp.c.llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = 1024;
    cparams.n_ubatch = 512;
    cparams.n_seq_max = 1;
    cparams.offload_kqv = true;

    // Threading optimization
    const cpu_count: i32 = @intCast(std.Thread.getCpuCount() catch 4);
    const final_threads = threads orelse cpu_count;
    cparams.n_threads = final_threads;
    cparams.n_threads_batch = final_threads;

    std.debug.print("Creating context (n_ctx={d}, n_threads={d})...\n", .{ cparams.n_ctx, cparams.n_threads });
    const ctx = try llama_cpp.Context.init(model, cparams);
    defer ctx.deinit();

    const vocab = model.vocab() orelse return error.ModelLoadFailed;
    const tmpl = model.chatTemplate();

    // Batch and Sampler
    var batch = llama_cpp.Batch.init(1024, 0, 1);
    defer batch.deinit();

    // Configurable sampler
    const sampler_config = llama_cpp.SamplerConfig{
        .temp = temp,
        .top_k = top_k,
        .top_p = top_p,
        .min_p = min_p,
        .seed = seed,
    };

    var grammar_z: ?[:0]u8 = null;
    defer if (grammar_z) |g| allocator.free(g);
    var grammar_root_z: ?[:0]u8 = null;
    defer if (grammar_root_z) |r| allocator.free(r);

    var sampler: llama_cpp.Sampler = undefined;
    if (grammar_path) |gp| {
        const grammar_bytes = try std.fs.cwd().readFileAlloc(allocator, gp, 4 * 1024 * 1024);
        defer allocator.free(grammar_bytes);

        grammar_z = try chat.dupeZ(allocator, grammar_bytes);
        grammar_root_z = try chat.dupeZ(allocator, grammar_root);

        sampler = try llama_cpp.Sampler.initWithConfigAndGrammar(sampler_config, vocab, grammar_z.?, grammar_root_z.?);
    } else {
        sampler = try llama_cpp.Sampler.initWithConfig(sampler_config);
    }
    defer sampler.deinit();

    var eval_tokens: std.ArrayList(llama_cpp.Token) = .empty;
    defer eval_tokens.deinit(allocator);

    var msgs: std.ArrayList(chat.Message) = .empty;
    if (load_chat_path) |load_path| {
        msgs = chat.loadJson(allocator, load_path) catch |err| switch (err) {
            error.FileNotFound => .empty,
            else => return err,
        };
    }
    defer chat.deinitMessages(allocator, &msgs);
    defer maybeSaveChat(allocator, save_chat_path, msgs.items);

    if (system_prompt) |sys| {
        try chat.setOrPrependSystemPrompt(allocator, &msgs, sys);
    }

    const stdout = std.fs.File.stdout().deprecatedWriter();
    const stdin_reader = std.fs.File.stdin().deprecatedReader();
    const stdin_is_tty = std.fs.File.stdin().isTty();
    const use_color = terminal.enableAnsiColors();

    if (prompt_mode) {
        if (user_prompt) |input| {
            // Allocate user message with errdefer for safety
            const user_z = try chat.dupeZ(allocator, input);
            errdefer allocator.free(user_z);
            try msgs.append(allocator, .{ .role = .user, .content = user_z });

            // Build prompt
            const ctx_reserve: usize = 256;
            const ctx_limit: usize = @as(usize, @intCast(cparams.n_ctx)) - ctx_reserve;
            var prompt = try buildPrompt(allocator, tmpl, vocab, msgs.items);
            if (prompt.tokens.len > ctx_limit) {
                std.debug.print("[Input too large for context]\n", .{});
                prompt.deinit(allocator);
                return;
            }
            defer prompt.deinit(allocator);

            const prompt_tokens = prompt.tokens;

            // Decode prompt
            var i: usize = 0;
            while (i < prompt_tokens.len) {
                batch.clear();
                const chunk_size = @min(prompt_tokens.len - i, @as(usize, @intCast(cparams.n_batch)));
                for (0..chunk_size) |j| {
                    try batch.add(prompt_tokens[i + j], @intCast(i + j), &[_]i32{0}, i + j == prompt_tokens.len - 1);
                }
                try ctx.decode(batch.handle);
                i += chunk_size;
            }
            try eval_tokens.appendSlice(allocator, prompt_tokens);

            // Generate response
            var n_gen: usize = 0;
            while (n_gen < 4096) : (n_gen += 1) {
                if (signal.shouldExit()) break;
                if (eval_tokens.items.len >= cparams.n_ctx) break;

                const token = sampler.sampleLast(ctx);
                if (llama_cpp.c.llama_vocab_is_eog(vocab, token)) break;

                // Use stack buffer for common case, heap for rare large tokens
                var piece_buf: [256]u8 = undefined;
                const n = llama_cpp.c.llama_token_to_piece(vocab, token, &piece_buf, piece_buf.len, 0, false);
                if (n > 0) {
                    const piece = piece_buf[0..@intCast(n)];
                    try stdout.writeAll(piece);
                } else if (n < 0) {
                    const actual_n: usize = @intCast(-n);
                    const large_buf = try allocator.alloc(u8, actual_n);
                    defer allocator.free(large_buf);
                    const n2 = llama_cpp.c.llama_token_to_piece(vocab, token, large_buf.ptr, @intCast(large_buf.len), 0, false);
                    if (n2 > 0) {
                        const piece = large_buf[0..@intCast(n2)];
                        try stdout.writeAll(piece);
                    }
                }

                batch.clear();
                try batch.add(token, @intCast(eval_tokens.items.len), &[_]i32{0}, true);
                try ctx.decode(batch.handle);
                try eval_tokens.append(allocator, token);
            }
            try stdout.writeAll("\n");
        }
        return;
    }

    try stdout.print("\nMLz Llama Chat Interface\n", .{});
    try stdout.print("Model context training length: {d}\n", .{model.nCtxTrain()});
    try stdout.print("Sampler: temp={d:.2}, top_k={d}, top_p={d:.2}, min_p={d:.2}, seed={d}, stream={}\n", .{
        sampler_config.temp,
        sampler_config.top_k,
        sampler_config.top_p,
        sampler_config.min_p,
        sampler_config.seed,
        stream,
    });
    if (grammar_path) |gp| {
        try stdout.print("Grammar: {s} (root: {s})\n", .{ gp, grammar_root });
    }
    if (system_prompt) |sys| {
        try stdout.print("System Prompt: {s}\n", .{sys});
    }
    try stdout.print("Commands: /clear, /reset, exit\n\n", .{});

    var line_buf: std.ArrayList(u8) = .empty;
    defer line_buf.deinit(allocator);

    while (true) {
        if (signal.shouldExit()) break;

        var user_input: []const u8 = "";
        try writeLabel(stdout, use_color, terminal.Ansi.green, "You");
        try stdout.writeAll(": ");
        line_buf.clearRetainingCapacity();
        stdin_reader.streamUntilDelimiter(line_buf.writer(allocator), '\n', null) catch |err| {
            if (err == error.EndOfStream) break;
            if (signal.shouldExit()) break;
            return err;
        };
        user_input = std.mem.trim(u8, line_buf.items, " \r\n");

        // In non-interactive runs (piped/redirected input), the terminal won't echo
        // the entered line (and therefore won't move to the next line). Emit a newline
        // so the assistant label doesn't appear on the same line as the prompt.
        if (!stdin_is_tty) try stdout.writeAll("\n");
        if (user_input.len == 0) continue;
        if (std.mem.eql(u8, user_input, "exit")) break;

        if (std.mem.eql(u8, user_input, "/clear") or std.mem.eql(u8, user_input, "/reset")) {
            chat.clearKeepSystem(allocator, &msgs);
            ctx.kvCacheSeqRm(0, 0, -1);
            eval_tokens.clearRetainingCapacity();
            sampler.reset();
            maybeSaveChat(allocator, save_chat_path, msgs.items);
            try writeDim(stdout, use_color);
            try stdout.writeAll("[conversation cleared]\n\n");
            try writeReset(stdout, use_color);
            continue;
        }

        // Allocate user message with errdefer for safety
        const user_z = try chat.dupeZ(allocator, user_input);
        errdefer allocator.free(user_z);
        try msgs.append(allocator, .{ .role = .user, .content = user_z });

        // Build prompt + tokenize, trimming oldest history if needed.
        const ctx_reserve: usize = 256;
        const ctx_limit: usize = @as(usize, @intCast(cparams.n_ctx)) - ctx_reserve;

        var prompt = try buildPrompt(allocator, tmpl, vocab, msgs.items);
        var dropped_any = false;
        while (prompt.tokens.len > ctx_limit) {
            prompt.deinit(allocator);
            if (!chat.dropOldestNonSystem(&msgs, allocator)) break;
            dropped_any = true;
            prompt = try buildPrompt(allocator, tmpl, vocab, msgs.items);
        }
        defer prompt.deinit(allocator);

        if (prompt.tokens.len > ctx_limit) {
            // Input is too large even after trimming history. Drop the last user message and continue.
            if (msgs.items.len > 0) {
                if (msgs.pop()) |removed| {
                    allocator.free(removed.content);
                }
            }
            try stdout.writeAll("[Input too large for context]\n");
            continue;
        }

        if (dropped_any) {
            // Sliding window changed the prompt substantially.
            ctx.kvCacheSeqRm(0, 0, -1);
            eval_tokens.clearRetainingCapacity();
        }

        const prompt_tokens = prompt.tokens;

        // Find common prefix with KV cache
        var common: usize = 0;
        while (common < eval_tokens.items.len and common < prompt_tokens.len and eval_tokens.items[common] == prompt_tokens[common]) : (common += 1) {}

        if (common < eval_tokens.items.len) {
            ctx.kvCacheSeqRm(0, @intCast(common), -1);
            eval_tokens.shrinkRetainingCapacity(common);
        }

        // Decode suffix
        if (common < prompt_tokens.len) {
            var i: usize = common;
            while (i < prompt_tokens.len) {
                batch.clear();
                const chunk_size = @min(prompt_tokens.len - i, @as(usize, @intCast(cparams.n_batch)));
                for (0..chunk_size) |j| {
                    try batch.add(prompt_tokens[i + j], @intCast(i + j), &[_]i32{0}, i + j == prompt_tokens.len - 1);
                }
                try ctx.decode(batch.handle);
                i += chunk_size;
            }
            try eval_tokens.appendSlice(allocator, prompt_tokens[common..]);
        }

        // Generate response
        try writeLabel(stdout, use_color, terminal.Ansi.cyan, "Assistant");
        try stdout.writeAll(": ");
        var assistant_buf: std.ArrayList(u8) = .empty;
        defer assistant_buf.deinit(allocator);

        sampler.reset();
        var gen_timer = try std.time.Timer.start();
        var ttft_ns: ?u64 = null;
        const prompt_token_count: usize = prompt_tokens.len;
        var completion_tokens: usize = 0;

        var n_gen: usize = 0;
        while (n_gen < 4096) : (n_gen += 1) {
            if (signal.shouldExit()) break;

            if (eval_tokens.items.len >= cparams.n_ctx) {
                try stdout.writeAll("\n[Context limit reached]\n");
                break;
            }

            const token = sampler.sampleLast(ctx);
            if (llama_cpp.c.llama_vocab_is_eog(vocab, token)) break;

            completion_tokens += 1;
            if (ttft_ns == null) ttft_ns = gen_timer.read();

            // Use stack buffer for common case, heap for rare large tokens
            var piece_buf: [256]u8 = undefined;
            const n = llama_cpp.c.llama_token_to_piece(vocab, token, &piece_buf, piece_buf.len, 0, false);
            if (n > 0) {
                const piece = piece_buf[0..@intCast(n)];
                if (stream) try stdout.writeAll(piece);
                try assistant_buf.appendSlice(allocator, piece);
            } else if (n < 0) {
                // Buffer too small, use heap
                const actual_n: usize = @intCast(-n);
                const large_buf = try allocator.alloc(u8, actual_n);
                defer allocator.free(large_buf);
                const n2 = llama_cpp.c.llama_token_to_piece(vocab, token, large_buf.ptr, @intCast(large_buf.len), 0, false);
                if (n2 > 0) {
                    const piece = large_buf[0..@intCast(n2)];
                    if (stream) try stdout.writeAll(piece);
                    try assistant_buf.appendSlice(allocator, piece);
                }
            }

            batch.clear();
            try batch.add(token, @intCast(eval_tokens.items.len), &[_]i32{0}, true);
            try ctx.decode(batch.handle);
            try eval_tokens.append(allocator, token);
        }
        if (!stream) try stdout.writeAll(assistant_buf.items);
        try stdout.writeAll("\n\n");

        const gen_total_ns: u64 = gen_timer.read();
        const ctx_total: usize = @intCast(cparams.n_ctx);
        const ctx_used: usize = eval_tokens.items.len;
        const ctx_remaining: usize = if (ctx_used < ctx_total) ctx_total - ctx_used else 0;
        try printTurnStats(
            stdout,
            use_color,
            prompt_token_count,
            completion_tokens,
            ctx_remaining,
            ttft_ns,
            gen_total_ns,
        );

        // Allocate assistant message with errdefer for safety
        const asst_z = try chat.dupeZ(allocator, assistant_buf.items);
        errdefer allocator.free(asst_z);
        try msgs.append(allocator, .{ .role = .assistant, .content = asst_z });

        maybeSaveChat(allocator, save_chat_path, msgs.items);
    }
}

const Prompt = struct {
    formatted: []u8,
    tokens: []llama_cpp.Token,

    fn deinit(self: Prompt, allocator: std.mem.Allocator) void {
        allocator.free(self.formatted);
        allocator.free(self.tokens);
    }
};

fn buildPrompt(
    allocator: std.mem.Allocator,
    tmpl: ?[*:0]const u8,
    vocab: ?*const llama_cpp.c.llama_vocab,
    msgs: []const chat.Message,
) !Prompt {
    const chat_msgs = try allocator.alloc(llama_cpp.c.llama_chat_message, msgs.len);
    defer allocator.free(chat_msgs);

    for (msgs, 0..) |m, i| {
        chat_msgs[i] = .{
            // NOTE: llama.cpp expects a null-terminated role string.
            .role = switch (m.role) {
                .system => "system",
                .user => "user",
                .assistant => "assistant",
            },
            .content = @as([*:0]const u8, m.content.ptr),
        };
    }

    const formatted = try llama_cpp.applyChatTemplate(allocator, tmpl, chat_msgs, true);
    errdefer allocator.free(formatted);

    const tokens = try llama_cpp.tokenize(allocator, vocab, formatted, true, true);
    errdefer allocator.free(tokens);

    return .{ .formatted = formatted, .tokens = tokens };
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
