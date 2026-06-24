//! Text embeddings via a model loaded in embedding mode (mean pooling).
//! Backs the `/v1/embeddings` endpoint. Separate from the chat Engine because a
//! context must be created with `embeddings = true` + a pooling type.

const std = @import("std");
const llama = @import("llama_cpp.zig");

pub const Embedder = struct {
    allocator: std.mem.Allocator,
    model: llama.Model,
    ctx: llama.Context,
    vocab: ?*const llama.c.llama_vocab,
    n_embd: usize,
    n_ctx: u32,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, n_ctx: u32, threads: ?i32) !Embedder {
        const path_z = try llama.dupeZ(allocator, model_path);
        defer allocator.free(path_z);

        var mparams = llama.c.llama_model_default_params();
        mparams.use_mmap = true;
        const model = try llama.Model.load(path_z, mparams);
        errdefer model.deinit();

        const cpu_count: i32 = @intCast(std.Thread.getCpuCount() catch 4);
        const eff_ctx: u32 = if (n_ctx == 0) blk: {
            const train = model.nCtxTrain();
            break :blk if (train > 0) @min(@as(u32, @intCast(train)), 8192) else 2048;
        } else n_ctx;

        const cparams = llama.embeddingContextParams(eff_ctx, threads orelse cpu_count);
        const ctx = try llama.Context.init(model, cparams);
        errdefer ctx.deinit();

        const n_embd: usize = @intCast(model.nEmbd());
        if (n_embd == 0) return error.NotAnEmbeddingModel;

        return .{
            .allocator = allocator,
            .model = model,
            .ctx = ctx,
            .vocab = model.vocab(),
            .n_embd = n_embd,
            .n_ctx = eff_ctx,
        };
    }

    pub fn deinit(self: *Embedder) void {
        self.ctx.deinit();
        self.model.deinit();
    }

    /// Embed one string -> L2-normalised vector of length n_embd. Caller frees.
    pub fn embed(self: *Embedder, allocator: std.mem.Allocator, text: []const u8) ![]f32 {
        const tokens = try llama.tokenize(self.allocator, self.vocab, text, true, false);
        defer self.allocator.free(tokens);
        if (tokens.len == 0) return error.EmptyInput;

        const n_use: usize = @min(tokens.len, @as(usize, self.n_ctx));

        // Fresh pooled embedding per call: clear the KV for seq 0 first.
        _ = self.ctx.kvCacheSeqRm(0, -1, -1);

        const batch = llama.c.llama_batch_get_one(tokens.ptr, @intCast(n_use));
        try self.ctx.decode(batch);

        const emb = self.ctx.embeddingsSeq(0) orelse return error.NoEmbeddings;
        const out = try allocator.alloc(f32, self.n_embd);
        errdefer allocator.free(out);
        @memcpy(out, emb[0..self.n_embd]);

        // L2 normalise (cosine-ready, matches common embedding server behaviour).
        var sum: f64 = 0;
        for (out) |v| sum += @as(f64, v) * @as(f64, v);
        if (sum > 0) {
            const inv: f32 = @floatCast(1.0 / @sqrt(sum));
            for (out) |*v| v.* *= inv;
        }
        return out;
    }
};

/// Lazily-loaded, single-model embedding service. Loads the requested model on
/// first use and reloads if a different model is requested. Thread-safe.
pub const EmbeddingService = struct {
    allocator: std.mem.Allocator,
    n_ctx: u32,
    threads: ?i32,
    mutex: std.Thread.Mutex = .{},
    current: ?Embedder = null,
    current_path: ?[]u8 = null,

    pub fn init(allocator: std.mem.Allocator, n_ctx: u32, threads: ?i32) EmbeddingService {
        return .{ .allocator = allocator, .n_ctx = n_ctx, .threads = threads };
    }

    pub fn deinit(self: *EmbeddingService) void {
        if (self.current) |*e| e.deinit();
        if (self.current_path) |p| self.allocator.free(p);
    }

    /// Embed `text` using the model at `model_path` (loaded/reused). Caller frees.
    pub fn embed(self: *EmbeddingService, allocator: std.mem.Allocator, model_path: []const u8, text: []const u8) ![]f32 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const need_reload = self.current_path == null or !std.mem.eql(u8, self.current_path.?, model_path);
        if (need_reload) {
            if (self.current) |*e| e.deinit();
            self.current = null;
            if (self.current_path) |p| self.allocator.free(p);
            self.current_path = null;
            self.current = try Embedder.init(self.allocator, model_path, self.n_ctx, self.threads);
            self.current_path = try self.allocator.dupe(u8, model_path);
        }
        return self.current.?.embed(allocator, text);
    }

    pub fn nEmbd(self: *EmbeddingService) ?usize {
        return if (self.current) |e| e.n_embd else null;
    }
};
