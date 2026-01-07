const std = @import("std");
const llama_cpp = @import("llama_cpp.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <model_path> [prompt]\n", .{args[0]});
        return;
    }

    const path = args[1];
    const initial_prompt = if (args.len > 2) args[2] else null;

    std.debug.print("Initializing llama.cpp backend...\n", .{});
    var backend = llama_cpp.Backend.init();
    defer backend.deinit();
    std.debug.print("System Info: {s}\n", .{llama_cpp.systemInfo()});

    const path_z = try llama_cpp.dupeZ(allocator, path);
    defer allocator.free(path_z);

    // Optimized model parameters
    var mparams = llama_cpp.c.llama_model_default_params();
    mparams.n_gpu_layers = 999; // Offload as much as possible to GPU
    mparams.use_mmap = true; // Use memory mapping for faster loading and less RAM
    mparams.use_mlock = false; // Don't pin memory by default

    std.debug.print("Loading model: {s}...\n", .{path});
    var model = try llama_cpp.Model.load(path_z, mparams);
    defer model.deinit();

    // Optimized context parameters
    var cparams = llama_cpp.c.llama_context_default_params();
    cparams.n_ctx = 4096; // Sufficient context for most tasks
    cparams.n_batch = 1024; // Larger batch for prompt processing
    cparams.n_ubatch = 512; // Micro-batch size
    cparams.n_seq_max = 1;
    cparams.offload_kqv = true; // Keep KV cache on GPU if possible

    // Threading optimization
    const cpu_count: i32 = @intCast(std.Thread.getCpuCount() catch 4);
    cparams.n_threads = cpu_count;
    cparams.n_threads_batch = cpu_count;

    std.debug.print("Creating context (n_ctx={d}, n_threads={d})...\n", .{ cparams.n_ctx, cparams.n_threads });
    var ctx = try llama_cpp.Context.init(model, cparams);
    defer ctx.deinit();

    const vocab = model.vocab() orelse return error.ModelLoadFailed;
    const tmpl = model.chatTemplate();

    // Batch and Sampler
    var batch = llama_cpp.Batch.init(1024, 0, 1);
    defer batch.deinit();

    const seed: u32 = @truncate(@as(u128, @intCast(std.time.nanoTimestamp())));
    var sampler = llama_cpp.Sampler.initAdvanced(0.8, 40, 0.95, seed);
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
    try stdout.print("Type 'exit' to quit.\n\n", .{});

    var line_buf: std.ArrayList(u8) = .empty;
    defer line_buf.deinit(allocator);

    while (true) {
        var user_input: []const u8 = "";
        if (initial_prompt != null and msgs.items.len == 0) {
            user_input = initial_prompt.?;
            try stdout.print("You: {s}\n", .{user_input});
        } else {
            try stdout.print("You: ", .{});
            line_buf.clearRetainingCapacity();
            stdin_reader.streamUntilDelimiter(line_buf.writer(allocator), '\n', null) catch |err| {
                if (err == error.EndOfStream) break else return err;
            };
            user_input = std.mem.trim(u8, line_buf.items, " \r\n");
            if (user_input.len == 0) continue;
            if (std.mem.eql(u8, user_input, "exit")) break;
        }

        const user_z = try llama_cpp.dupeZ(allocator, user_input);
        try msgs.append(allocator, .{ .role = .user, .content = user_z });

        // Apply chat template
        var chat_msgs = try allocator.alloc(llama_cpp.c.llama_chat_message, msgs.items.len);
        defer allocator.free(chat_msgs);
        for (msgs.items, 0..) |m, i| {
            chat_msgs[i] = .{
                .role = switch (m.role) {
                    .user => "user",
                    .assistant => "assistant",
                },
                .content = m.content.ptr,
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
                    batch.add(prompt_tokens[i + j], @intCast(i + j), &[_]i32{0}, i + j == prompt_tokens.len - 1);
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

            const token = sampler.sample(ctx, -1);
            if (llama_cpp.c.llama_vocab_is_eog(vocab, token)) break;

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
            batch.add(token, @intCast(eval_tokens.items.len), &[_]i32{0}, true);
            try ctx.decode(batch.handle);
            try eval_tokens.append(allocator, token);
        }
        try stdout.writeAll("\n\n");

        const asst_z = try llama_cpp.dupeZ(allocator, assistant_buf.items);
        try msgs.append(allocator, .{ .role = .assistant, .content = asst_z });
    }
}

const Role = enum { user, assistant };
const StoredMsg = struct {
    role: Role,
    content: [:0]u8,
};
