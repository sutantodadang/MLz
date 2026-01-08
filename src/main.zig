const std = @import("std");
const llama_cpp = @import("llama_cpp.zig");

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

    var arg_idx: usize = 1;
    while (arg_idx < args.len) : (arg_idx += 1) {
        const arg = args[arg_idx];
        if (std.mem.eql(u8, arg, "--temp")) {
            arg_idx += 1;
            if (arg_idx < args.len) temp = std.fmt.parseFloat(f32, args[arg_idx]) catch temp;
        } else if (std.mem.eql(u8, arg, "--top-k")) {
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
            \\  --temp <float>       Temperature (default: 0.8)
            \\  --top-k <int>         Top-K (default: 40)
            \\  --top-p <float>       Top-P (default: 0.95)
            \\  --min-p <float>       Min-P (default: 0.05)
            \\  --seed <int>          Seed (default: 42)
            \\  --ctx <int>           Context size (default: 4096)
            \\  --ngl <int>           GPU layers (default: 999)
            \\  --threads <int>       Number of threads (default: CPU count)
            \\
        , .{args[0]});
        return;
    }

    const model_path = path.?;

    std.debug.print("Initializing llama.cpp backend...\n", .{});
    const backend = llama_cpp.Backend.init();
    defer backend.deinit();

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
    const sampler = try llama_cpp.Sampler.initWithConfig(sampler_config);
    defer sampler.deinit();

    var eval_tokens: std.ArrayList(llama_cpp.Token) = .empty;
    defer eval_tokens.deinit(allocator);

    var msgs: std.ArrayList(StoredMsg) = .empty;
    defer {
        for (msgs.items) |m| allocator.free(m.content);
        msgs.deinit(allocator);
    }

    const stdout = std.fs.File.stdout().deprecatedWriter();
    const stdin_reader = std.fs.File.stdin().deprecatedReader();

    try stdout.print("\nMLz Llama Chat Interface\n", .{});
    try stdout.print("Model context training length: {d}\n", .{model.nCtxTrain()});
    try stdout.print("Sampler: temp={d:.2}, top_k={d}, top_p={d:.2}, min_p={d:.2}, seed={d}\n", .{
        sampler_config.temp,
        sampler_config.top_k,
        sampler_config.top_p,
        sampler_config.min_p,
        sampler_config.seed,
    });
    try stdout.print("Type 'exit' to quit.\n\n", .{});

    var line_buf: std.ArrayList(u8) = .empty;
    defer line_buf.deinit(allocator);

    while (true) {
        var user_input: []const u8 = "";
        try stdout.print("You: ", .{});
        line_buf.clearRetainingCapacity();
        stdin_reader.streamUntilDelimiter(line_buf.writer(allocator), '\n', null) catch |err| {
            if (err == error.EndOfStream) break else return err;
        };
        user_input = std.mem.trim(u8, line_buf.items, " \r\n");
        if (user_input.len == 0) continue;
        if (std.mem.eql(u8, user_input, "exit")) break;

        // Allocate user message with errdefer for safety
        const user_z = try llama_cpp.dupeZ(allocator, user_input);
        errdefer allocator.free(user_z);
        try msgs.append(allocator, .{ .role = .user, .content = user_z });

        // Apply chat template
        // IMPORTANT: chat_msgs holds pointers into msgs.items[*].content
        // These pointers remain valid because we don't modify msgs until after
        // applyChatTemplate returns and we free chat_msgs.
        const chat_msgs = try allocator.alloc(llama_cpp.c.llama_chat_message, msgs.items.len);
        defer allocator.free(chat_msgs);

        for (msgs.items, 0..) |m, i| {
            chat_msgs[i] = .{
                .role = switch (m.role) {
                    .user => "user",
                    .assistant => "assistant",
                },
                // Explicit sentinel cast ensures compile-time check that content is null-terminated
                .content = @as([*:0]const u8, m.content.ptr),
            };
        }

        const formatted = try llama_cpp.applyChatTemplate(allocator, tmpl, chat_msgs, true);
        defer allocator.free(formatted);

        // Tokenize
        const prompt_tokens = try llama_cpp.tokenize(allocator, vocab, formatted, true, true);
        defer allocator.free(prompt_tokens);

        // Context management: check if we need to remove old history
        if (prompt_tokens.len > cparams.n_ctx - 256) {
            // Very basic: just clear and restart if too long
            ctx.kvCacheSeqRm(0, 0, -1);
            eval_tokens.clearRetainingCapacity();
        }

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
        try stdout.print("Assistant: ", .{});
        var assistant_buf: std.ArrayList(u8) = .empty;
        defer assistant_buf.deinit(allocator);

        var n_gen: usize = 0;
        while (n_gen < 4096) : (n_gen += 1) {
            if (eval_tokens.items.len >= cparams.n_ctx) {
                try stdout.writeAll("\n[Context limit reached]\n");
                break;
            }

            const token = sampler.sampleLast(ctx);
            if (llama_cpp.c.llama_vocab_is_eog(vocab, token)) break;

            // Use stack buffer for common case, heap for rare large tokens
            var piece_buf: [256]u8 = undefined;
            const n = llama_cpp.c.llama_token_to_piece(vocab, token, &piece_buf, piece_buf.len, 0, false);
            if (n > 0) {
                const piece = piece_buf[0..@intCast(n)];
                try stdout.writeAll(piece);
                try assistant_buf.appendSlice(allocator, piece);
            } else if (n < 0) {
                // Buffer too small, use heap
                const actual_n: usize = @intCast(-n);
                const large_buf = try allocator.alloc(u8, actual_n);
                defer allocator.free(large_buf);
                const n2 = llama_cpp.c.llama_token_to_piece(vocab, token, large_buf.ptr, @intCast(large_buf.len), 0, false);
                if (n2 > 0) {
                    const piece = large_buf[0..@intCast(n2)];
                    try stdout.writeAll(piece);
                    try assistant_buf.appendSlice(allocator, piece);
                }
            }

            batch.clear();
            try batch.add(token, @intCast(eval_tokens.items.len), &[_]i32{0}, true);
            try ctx.decode(batch.handle);
            try eval_tokens.append(allocator, token);
        }
        try stdout.writeAll("\n\n");

        // Allocate assistant message with errdefer for safety
        const asst_z = try llama_cpp.dupeZ(allocator, assistant_buf.items);
        errdefer allocator.free(asst_z);
        try msgs.append(allocator, .{ .role = .assistant, .content = asst_z });
    }
}

const Role = enum { user, assistant };
const StoredMsg = struct {
    role: Role,
    content: [:0]u8,
};
