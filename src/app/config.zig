const std = @import("std");

pub const starter_toml: []const u8 =
    \\# MLz configuration. Copy to mlz.toml (auto-loaded from cwd) or pass --config <path>.
    \\# Precedence: built-in defaults < this file < MLZ_* env vars < CLI flags.
    \\
    \\[model]
    \\path = "models/model.gguf"
    \\n_ctx = 4096
    \\n_gpu_layers = "auto"   # "auto" = use default offload (999); or an integer
    \\threads = "auto"        # "auto" = physical core count; or an integer
    \\
    \\[serve]
    \\enabled = false
    \\host = "127.0.0.1"
    \\port = 8080
    \\# api_key = "secret"
    \\
    \\[sampling]
    \\temp = 0.8
    \\top_k = 40
    \\top_p = 0.95
    \\min_p = 0.05
    \\seed = 42
    \\
    \\[chat]
    \\stream = true
    \\# system = "You are a helpful assistant."
    \\# template = "gemma"
    \\# grammar = "grammar.gbnf"
    \\grammar_root = "root"
    \\
    \\[speculative]
    \\# draft_model = "models/draft.gguf"
    \\
    \\# Official GGML bounded weight residency. Requires a build made with
    \\# -Dggml-residency-hooks=true. This limits active mapped immutable
    \\# weights, not total process RSS/KV/workspace memory.
    \\[residency]
    \\enabled = false
    \\backend = "ggml"
    \\weight_budget_mib = 256
    \\
;

fn parseBool(s: []const u8) bool {
    return std.mem.eql(u8, s, "true") or std.mem.eql(u8, s, "1");
}

pub const Config = struct {
    // Model Config
    model_path: []const u8 = "", // Set during parsing
    n_ctx: u32 = 4096,
    n_gpu_layers: i32 = 999,
    threads: ?i32 = null,

    // Sampling Config
    temp: f32 = 0.8,
    top_k: i32 = 40,
    top_p: f32 = 0.95,
    min_p: f32 = 0.05,
    seed: u32 = 42,

    // Chat / Interaction Config
    stream: bool = true,
    system_prompt: ?[]const u8 = null,
    save_chat_path: ?[]const u8 = null,
    load_chat_path: ?[]const u8 = null,
    grammar_path: ?[]const u8 = null,
    grammar_root: []const u8 = "root",
    chat_template: ?[]const u8 = null,

    // One-shot prompt mode
    prompt_mode: bool = false,
    user_prompt: ?[]const u8 = null,
    n_predict: ?usize = null, // --n-predict : max tokens to generate in prompt mode

    // Speculative Decoding
    draft_model_path: ?[]const u8 = null,

    // Server Config
    server_mode: bool = false,
    server_host: []const u8 = "127.0.0.1",
    server_port: u16 = 8080,
    server_api_key: ?[]const u8 = null,
    max_concurrent: u32 = 1,
    prefix_cache: bool = true,

    // Official GGML bounded-residency backend for the normal model path.
    residency_enabled: bool = false,
    residency_backend: []const u8 = "ggml",
    residency_weight_budget_mib: usize = 256,
    /// Optional hard limit for KV/recurrent state, graph workspace, and logits,
    /// preflighted from llama.cpp's simulated (`no_alloc`) allocation sizes.
    residency_state_budget_mib: ?usize = null,

    // Custom SIMD backend runtime controls (consumed before model load to set
    // env vars read by ggml_simd_hook.cpp).  Defaults preserve the build-time
    // behaviour: SIMD on if compiled with -Dsimd-backend=true, off otherwise.
    no_simd: bool = false, // --no-simd : sets MLZ_SIMD=0
    simd_trace: bool = false, // --simd-trace : sets MLZ_SIMD_TRACE=1
    simd_flash_attn: bool = false, // --simd-flash-attn : opt-in for hooked FA

    // prompt memory management
    _allocated_prompt: ?[]u8 = null,

    // Owns strings parsed from TOML file / env vars (CLI strings borrow argv).
    _arena: ?*std.heap.ArenaAllocator = null,
    print_config: bool = false, // --print-config: dump resolved config and exit
    init_config: bool = false, // --init: write starter mlz.toml to cwd

    pub const ParseError = error{
        MissingModelPath,
        InvalidFloat,
        InvalidInt,
        InvalidResidencyBackend,
        ResidencyBudgetRequired,
        ResidencyRequiresCpu,
        ResidencyIncompatibleDraftModel,
        MissingArgument,
        OutOfMemory,
    };

    fn arenaAlloc(self: *Config, allocator: std.mem.Allocator) !std.mem.Allocator {
        if (self._arena == null) {
            const a = try allocator.create(std.heap.ArenaAllocator);
            a.* = std.heap.ArenaAllocator.init(allocator);
            self._arena = a;
        }
        return self._arena.?.allocator();
    }

    pub fn deinit(self: *const Config, allocator: std.mem.Allocator) void {
        if (self._allocated_prompt) |p| allocator.free(p);
        if (self._arena) |a| {
            a.deinit();
            allocator.destroy(a);
        }
    }

    pub fn parse(allocator: std.mem.Allocator, args: []const [:0]u8) !Config {
        var cfg = Config{};
        errdefer cfg.deinit(allocator);

        // Step 1: find --config path or default mlz.toml
        var config_file_path: ?[]const u8 = null;
        {
            var i: usize = 1;
            while (i < args.len) : (i += 1) {
                if (std.mem.eql(u8, args[i], "--config")) {
                    i += 1;
                    if (i < args.len) config_file_path = args[i];
                }
            }
        }

        // Step 2: load TOML file
        if (config_file_path) |path| {
            // explicit path — error if not found
            const file = try std.fs.cwd().openFile(path, .{});
            defer file.close();
            const text = try file.readToEndAlloc(allocator, 1 << 20);
            defer allocator.free(text);
            try cfg.applyToml(allocator, text);
        } else {
            // try default mlz.toml — silently ignore FileNotFound
            if (std.fs.cwd().openFile("mlz.toml", .{})) |file| {
                defer file.close();
                const text = file.readToEndAlloc(allocator, 1 << 20) catch |err| {
                    if (err == error.FileTooBig) return ParseError.OutOfMemory;
                    return err;
                };
                defer allocator.free(text);
                try cfg.applyToml(allocator, text);
            } else |err| {
                if (err != error.FileNotFound) return err;
            }
        }

        // Step 3: env vars
        try cfg.applyEnv(allocator);

        // Step 4: CLI args
        try cfg.applyArgs(allocator, args);

        // Step 5: validate
        if (cfg.model_path.len == 0) return ParseError.MissingModelPath;
        try cfg.validateResidency();
        return cfg;
    }

    fn validateResidency(self: *const Config) ParseError!void {
        if (!self.residency_enabled) return;
        if (!std.mem.eql(u8, self.residency_backend, "ggml")) {
            return ParseError.InvalidResidencyBackend;
        }
        if (self.residency_weight_budget_mib == 0) {
            return ParseError.ResidencyBudgetRequired;
        }
        // 999 is the application's "auto" default; residency resolves it to
        // CPU-only in Engine.init. Explicit GPU offload would bypass the limit.
        if (self.n_gpu_layers != 0 and self.n_gpu_layers != 999) return ParseError.ResidencyRequiresCpu;
        // The bridge and C registry are process-global and currently own one
        // backed model. Reject configurations which would load another model.
        if (self.draft_model_path != null) return ParseError.ResidencyIncompatibleDraftModel;
    }

    fn applyArgs(self: *Config, allocator: std.mem.Allocator, args: []const [:0]u8) !void {
        var path_set = self.model_path.len > 0;

        var i: usize = 1;
        while (i < args.len) : (i += 1) {
            const arg = args[i];

            if (std.mem.eql(u8, arg, "--temp")) {
                self.temp = try parseNextFloat(&i, args);
            } else if (std.mem.eql(u8, arg, "--prompt")) {
                i += 1;
                if (i < args.len) {
                    var prompt_parts = std.ArrayList([]const u8){};
                    defer prompt_parts.deinit(allocator);

                    try prompt_parts.append(allocator, args[i]);

                    // Greedily consume subsequent arguments if they don't look like flags
                    while (i + 1 < args.len) {
                        const next_arg = args[i + 1];
                        if (std.mem.startsWith(u8, next_arg, "-")) break;
                        i += 1;
                        try prompt_parts.append(allocator, args[i]);
                    }

                    if (prompt_parts.items.len == 1) {
                        self.user_prompt = prompt_parts.items[0];
                    } else {
                        const joined = try std.mem.join(allocator, " ", prompt_parts.items);
                        self._allocated_prompt = joined;
                        self.user_prompt = joined;
                    }
                    self.prompt_mode = true;
                } else {
                    return ParseError.MissingArgument;
                }
            } else if (std.mem.eql(u8, arg, "--top-k")) {
                self.top_k = try parseNextInt(i32, &i, args);
            } else if (std.mem.eql(u8, arg, "--top-p")) {
                self.top_p = try parseNextFloat(&i, args);
            } else if (std.mem.eql(u8, arg, "--min-p")) {
                self.min_p = try parseNextFloat(&i, args);
            } else if (std.mem.eql(u8, arg, "--seed")) {
                self.seed = try parseNextInt(u32, &i, args);
            } else if (std.mem.eql(u8, arg, "--ctx")) {
                self.n_ctx = try parseNextInt(u32, &i, args);
            } else if (std.mem.eql(u8, arg, "--ngl")) {
                self.n_gpu_layers = try parseNextInt(i32, &i, args);
            } else if (std.mem.eql(u8, arg, "--threads")) {
                self.threads = try parseNextInt(i32, &i, args);
            } else if (std.mem.eql(u8, arg, "--stream")) {
                i += 1;
                if (i < args.len) {
                    const val = args[i];
                    self.stream = std.mem.eql(u8, val, "true") or std.mem.eql(u8, val, "1");
                }
            } else if (std.mem.eql(u8, arg, "--system")) {
                self.system_prompt = try getNextArg(&i, args);
            } else if (std.mem.eql(u8, arg, "--save-chat")) {
                self.save_chat_path = try getNextArg(&i, args);
            } else if (std.mem.eql(u8, arg, "--load-chat")) {
                self.load_chat_path = try getNextArg(&i, args);
            } else if (std.mem.eql(u8, arg, "--grammar")) {
                self.grammar_path = try getNextArg(&i, args);
            } else if (std.mem.eql(u8, arg, "--grammar-root")) {
                self.grammar_root = try getNextArg(&i, args) orelse "root";
            } else if (std.mem.eql(u8, arg, "--chat-template")) {
                self.chat_template = try getNextArg(&i, args);
            } else if (std.mem.eql(u8, arg, "--draft-model")) {
                self.draft_model_path = try getNextArg(&i, args);
            } else if (std.mem.eql(u8, arg, "--server")) {
                self.server_mode = true;
            } else if (std.mem.eql(u8, arg, "--host")) {
                self.server_host = try getNextArg(&i, args) orelse "127.0.0.1";
            } else if (std.mem.eql(u8, arg, "--port")) {
                self.server_port = try parseNextInt(u16, &i, args);
            } else if (std.mem.eql(u8, arg, "--api-key")) {
                self.server_api_key = try getNextArg(&i, args);
            } else if (std.mem.eql(u8, arg, "--max-concurrent")) {
                self.max_concurrent = try parseNextInt(u32, &i, args);
            } else if (std.mem.eql(u8, arg, "--prefix-cache")) {
                self.prefix_cache = true;
            } else if (std.mem.eql(u8, arg, "--residency")) {
                self.residency_enabled = true;
            } else if (std.mem.eql(u8, arg, "--residency-backend")) {
                self.residency_backend = try getNextArg(&i, args) orelse return ParseError.MissingArgument;
            } else if (std.mem.eql(u8, arg, "--weight-budget-mib")) {
                self.residency_weight_budget_mib = try parseNextInt(usize, &i, args);
            } else if (std.mem.eql(u8, arg, "--state-budget-mib")) {
                self.residency_state_budget_mib = try parseNextInt(usize, &i, args);
            } else if (std.mem.eql(u8, arg, "--no-prefix-cache")) {
                self.prefix_cache = false;
            } else if (std.mem.eql(u8, arg, "--no-simd")) {
                self.no_simd = true;
            } else if (std.mem.eql(u8, arg, "--n-predict")) {
                const v = try parseNextInt(usize, &i, args);
                // Clamp to a sane upper bound (1 MiB tokens) to prevent
                // accidental near-infinite generation from typos like
                // `--n-predict 18446744073709551615`.  Mirrors the
                // openai.zig max_tokens guard.
                const n_predict_max: usize = 1 << 20;
                if (v == 0) return ParseError.InvalidInt;
                self.n_predict = if (v > n_predict_max) n_predict_max else v;
            } else if (std.mem.eql(u8, arg, "--simd-trace")) {
                self.simd_trace = true;
            } else if (std.mem.eql(u8, arg, "--simd-flash-attn")) {
                self.simd_flash_attn = true;
            } else if (std.mem.eql(u8, arg, "--config")) {
                i += 1; // consume value (already handled in parse)
            } else if (std.mem.eql(u8, arg, "--print-config")) {
                self.print_config = true;
            } else if (std.mem.eql(u8, arg, "--init")) {
                self.init_config = true;
            } else if (std.mem.startsWith(u8, arg, "--")) {
                std.log.warn("Unknown argument: {s}", .{arg});
            } else {
                if (!path_set) {
                    self.model_path = arg;
                    path_set = true;
                }
            }
        }
    }

    fn applyToml(self: *Config, allocator: std.mem.Allocator, text: []const u8) !void {
        const arena = try self.arenaAlloc(allocator);
        var section: []const u8 = "";
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |raw_line| {
            var line = std.mem.trim(u8, raw_line, " \t\r");
            if (line.len == 0 or line[0] == '#') continue;
            // section header
            if (line[0] == '[') {
                const end = std.mem.indexOfScalar(u8, line, ']') orelse continue;
                section = std.mem.trim(u8, line[1..end], " \t");
                continue;
            }
            // key = value
            const eq = std.mem.indexOfScalar(u8, line, '=') orelse continue;
            const key = std.mem.trim(u8, line[0..eq], " \t");
            var val = std.mem.trim(u8, line[eq + 1 ..], " \t");
            // strip inline comment on unquoted values
            if (val.len > 0 and val[0] != '"') {
                if (std.mem.indexOfScalar(u8, val, '#')) |ci| {
                    val = std.mem.trim(u8, val[0..ci], " \t");
                }
            }
            // strip surrounding quotes
            const is_quoted = val.len >= 2 and val[0] == '"' and val[val.len - 1] == '"';
            const str_val = if (is_quoted) val[1 .. val.len - 1] else val;

            // dispatch by section.key
            if (std.mem.eql(u8, section, "model")) {
                if (std.mem.eql(u8, key, "path")) {
                    self.model_path = try arena.dupe(u8, str_val);
                } else if (std.mem.eql(u8, key, "n_ctx")) {
                    if (std.mem.eql(u8, str_val, "auto")) {
                        self.n_ctx = 0;
                    } else {
                        self.n_ctx = std.fmt.parseInt(u32, str_val, 10) catch return ParseError.InvalidInt;
                    }
                } else if (std.mem.eql(u8, key, "n_gpu_layers")) {
                    if (!std.mem.eql(u8, str_val, "auto")) {
                        self.n_gpu_layers = std.fmt.parseInt(i32, str_val, 10) catch return ParseError.InvalidInt;
                    }
                } else if (std.mem.eql(u8, key, "threads")) {
                    if (!std.mem.eql(u8, str_val, "auto")) {
                        self.threads = std.fmt.parseInt(i32, str_val, 10) catch return ParseError.InvalidInt;
                    }
                } else {
                    std.log.warn("mlz.toml: unknown key model.{s}", .{key});
                }
            } else if (std.mem.eql(u8, section, "serve")) {
                if (std.mem.eql(u8, key, "host")) {
                    self.server_host = try arena.dupe(u8, str_val);
                } else if (std.mem.eql(u8, key, "port")) {
                    self.server_port = std.fmt.parseInt(u16, str_val, 10) catch return ParseError.InvalidInt;
                } else if (std.mem.eql(u8, key, "api_key")) {
                    self.server_api_key = try arena.dupe(u8, str_val);
                } else if (std.mem.eql(u8, key, "enabled")) {
                    self.server_mode = parseBool(str_val);
                } else if (std.mem.eql(u8, key, "max_concurrent")) {
                    self.max_concurrent = std.fmt.parseInt(u32, str_val, 10) catch return ParseError.InvalidInt;
                } else if (std.mem.eql(u8, key, "prefix_cache")) {
                    self.prefix_cache = parseBool(str_val);
                } else {
                    std.log.warn("mlz.toml: unknown key serve.{s}", .{key});
                }
            } else if (std.mem.eql(u8, section, "residency")) {
                if (std.mem.eql(u8, key, "enabled")) {
                    self.residency_enabled = parseBool(str_val);
                } else if (std.mem.eql(u8, key, "backend")) {
                    self.residency_backend = try arena.dupe(u8, str_val);
                } else if (std.mem.eql(u8, key, "weight_budget_mib")) {
                    self.residency_weight_budget_mib = std.fmt.parseInt(usize, str_val, 10) catch return ParseError.InvalidInt;
                } else if (std.mem.eql(u8, key, "state_budget_mib")) {
                    self.residency_state_budget_mib = std.fmt.parseInt(usize, str_val, 10) catch return ParseError.InvalidInt;
                } else {
                    std.log.warn("mlz.toml: unknown key residency.{s}", .{key});
                }
            } else if (std.mem.eql(u8, section, "sampling")) {
                if (std.mem.eql(u8, key, "temp")) {
                    self.temp = std.fmt.parseFloat(f32, str_val) catch return ParseError.InvalidFloat;
                } else if (std.mem.eql(u8, key, "top_k")) {
                    self.top_k = std.fmt.parseInt(i32, str_val, 10) catch return ParseError.InvalidInt;
                } else if (std.mem.eql(u8, key, "top_p")) {
                    self.top_p = std.fmt.parseFloat(f32, str_val) catch return ParseError.InvalidFloat;
                } else if (std.mem.eql(u8, key, "min_p")) {
                    self.min_p = std.fmt.parseFloat(f32, str_val) catch return ParseError.InvalidFloat;
                } else if (std.mem.eql(u8, key, "seed")) {
                    self.seed = std.fmt.parseInt(u32, str_val, 10) catch return ParseError.InvalidInt;
                } else {
                    std.log.warn("mlz.toml: unknown key sampling.{s}", .{key});
                }
            } else if (std.mem.eql(u8, section, "speculative")) {
                if (std.mem.eql(u8, key, "draft_model")) {
                    self.draft_model_path = try arena.dupe(u8, str_val);
                } else {
                    std.log.warn("mlz.toml: unknown key speculative.{s}", .{key});
                }
            } else if (std.mem.eql(u8, section, "chat")) {
                if (std.mem.eql(u8, key, "system")) {
                    self.system_prompt = try arena.dupe(u8, str_val);
                } else if (std.mem.eql(u8, key, "template")) {
                    self.chat_template = try arena.dupe(u8, str_val);
                } else if (std.mem.eql(u8, key, "grammar")) {
                    self.grammar_path = try arena.dupe(u8, str_val);
                } else if (std.mem.eql(u8, key, "grammar_root")) {
                    self.grammar_root = try arena.dupe(u8, str_val);
                } else if (std.mem.eql(u8, key, "stream")) {
                    self.stream = parseBool(str_val);
                } else {
                    std.log.warn("mlz.toml: unknown key chat.{s}", .{key});
                }
            } else if (section.len > 0) {
                std.log.warn("mlz.toml: unknown section [{s}]", .{section});
            }
        }
    }

    fn applyEnv(self: *Config, allocator: std.mem.Allocator) !void {
        const arena = try self.arenaAlloc(allocator);

        // helper: get env var, dupe into arena, free owned copy
        const getStr = struct {
            fn call(a: std.mem.Allocator, ar: std.mem.Allocator, name: []const u8) !?[]const u8 {
                const owned = std.process.getEnvVarOwned(a, name) catch |err| {
                    if (err == error.EnvironmentVariableNotFound) return null;
                    return err;
                };
                defer a.free(owned);
                return try ar.dupe(u8, owned);
            }
        }.call;

        if (try getStr(allocator, arena, "MLZ_MODEL")) |v| self.model_path = v;

        if (std.process.getEnvVarOwned(allocator, "MLZ_N_CTX") catch null) |v| {
            defer allocator.free(v);
            if (std.mem.eql(u8, v, "auto")) {
                self.n_ctx = 0;
            } else {
                self.n_ctx = std.fmt.parseInt(u32, v, 10) catch return ParseError.InvalidInt;
            }
        }
        if (std.process.getEnvVarOwned(allocator, "MLZ_N_GPU_LAYERS") catch null) |v| {
            defer allocator.free(v);
            self.n_gpu_layers = std.fmt.parseInt(i32, v, 10) catch return ParseError.InvalidInt;
        }
        if (std.process.getEnvVarOwned(allocator, "MLZ_THREADS") catch null) |v| {
            defer allocator.free(v);
            self.threads = std.fmt.parseInt(i32, v, 10) catch return ParseError.InvalidInt;
        }
        if (try getStr(allocator, arena, "MLZ_HOST")) |v| self.server_host = v;
        if (std.process.getEnvVarOwned(allocator, "MLZ_PORT") catch null) |v| {
            defer allocator.free(v);
            self.server_port = std.fmt.parseInt(u16, v, 10) catch return ParseError.InvalidInt;
        }
        if (try getStr(allocator, arena, "MLZ_API_KEY")) |v| self.server_api_key = v;
        if (std.process.getEnvVarOwned(allocator, "MLZ_MAX_CONCURRENT") catch null) |v| {
            defer allocator.free(v);
            self.max_concurrent = std.fmt.parseInt(u32, v, 10) catch return ParseError.InvalidInt;
        }
        if (std.process.getEnvVarOwned(allocator, "MLZ_PREFIX_CACHE") catch null) |v| {
            defer allocator.free(v);
            self.prefix_cache = parseBool(v);
        }
        if (std.process.getEnvVarOwned(allocator, "MLZ_RESIDENCY_ENABLED") catch null) |v| {
            defer allocator.free(v);
            self.residency_enabled = parseBool(v);
        }
        if (try getStr(allocator, arena, "MLZ_RESIDENCY_BACKEND")) |v| self.residency_backend = v;
        if (std.process.getEnvVarOwned(allocator, "MLZ_WEIGHT_BUDGET_MIB") catch null) |v| {
            defer allocator.free(v);
            self.residency_weight_budget_mib = std.fmt.parseInt(usize, v, 10) catch return ParseError.InvalidInt;
        }
        if (std.process.getEnvVarOwned(allocator, "MLZ_STATE_BUDGET_MIB") catch null) |v| {
            defer allocator.free(v);
            self.residency_state_budget_mib = std.fmt.parseInt(usize, v, 10) catch return ParseError.InvalidInt;
        }
        if (std.process.getEnvVarOwned(allocator, "MLZ_TEMP") catch null) |v| {
            defer allocator.free(v);
            self.temp = std.fmt.parseFloat(f32, v) catch return ParseError.InvalidFloat;
        }
        if (std.process.getEnvVarOwned(allocator, "MLZ_TOP_K") catch null) |v| {
            defer allocator.free(v);
            self.top_k = std.fmt.parseInt(i32, v, 10) catch return ParseError.InvalidInt;
        }
        if (std.process.getEnvVarOwned(allocator, "MLZ_TOP_P") catch null) |v| {
            defer allocator.free(v);
            self.top_p = std.fmt.parseFloat(f32, v) catch return ParseError.InvalidFloat;
        }
        if (std.process.getEnvVarOwned(allocator, "MLZ_MIN_P") catch null) |v| {
            defer allocator.free(v);
            self.min_p = std.fmt.parseFloat(f32, v) catch return ParseError.InvalidFloat;
        }
        if (std.process.getEnvVarOwned(allocator, "MLZ_SEED") catch null) |v| {
            defer allocator.free(v);
            self.seed = std.fmt.parseInt(u32, v, 10) catch return ParseError.InvalidInt;
        }
    }

    pub fn dump(self: *const Config, writer: anytype) !void {
        try writer.print("[model]\n", .{});
        try writer.print("path = \"{s}\"\n", .{self.model_path});
        if (self.n_ctx == 0) {
            try writer.print("n_ctx = \"auto\"\n", .{});
        } else {
            try writer.print("n_ctx = {d}\n", .{self.n_ctx});
        }
        try writer.print("n_gpu_layers = {d}\n", .{self.n_gpu_layers});
        if (self.threads) |t| {
            try writer.print("threads = {d}\n", .{t});
        } else {
            try writer.print("threads = \"auto\"\n", .{});
        }
        try writer.print("\n[serve]\n", .{});
        try writer.print("enabled = {}\n", .{self.server_mode});
        try writer.print("host = \"{s}\"\n", .{self.server_host});
        try writer.print("port = {d}\n", .{self.server_port});
        try writer.print("max_concurrent = {d}\n", .{self.max_concurrent});
        try writer.print("prefix_cache = {}\n", .{self.prefix_cache});
        if (self.server_api_key) |k| {
            try writer.print("api_key = \"{s}\"\n", .{k});
        }
        try writer.print("\n[residency]\n", .{});
        try writer.print("enabled = {}\n", .{self.residency_enabled});
        try writer.print("backend = \"{s}\"\n", .{self.residency_backend});
        try writer.print("weight_budget_mib = {d}\n", .{self.residency_weight_budget_mib});
        if (self.residency_state_budget_mib) |limit| {
            try writer.print("state_budget_mib = {d}\n", .{limit});
        }
        try writer.print("\n[sampling]\n", .{});
        try writer.print("temp = {d:.4}\n", .{self.temp});
        try writer.print("top_k = {d}\n", .{self.top_k});
        try writer.print("top_p = {d:.4}\n", .{self.top_p});
        try writer.print("min_p = {d:.4}\n", .{self.min_p});
        try writer.print("seed = {d}\n", .{self.seed});
        try writer.print("\n[chat]\n", .{});
        try writer.print("stream = {}\n", .{self.stream});
        if (self.system_prompt) |s| try writer.print("system = \"{s}\"\n", .{s});
        if (self.chat_template) |t| try writer.print("template = \"{s}\"\n", .{t});
        if (self.grammar_path) |g| try writer.print("grammar = \"{s}\"\n", .{g});
        try writer.print("grammar_root = \"{s}\"\n", .{self.grammar_root});
        try writer.print("\n[speculative]\n", .{});
        if (self.draft_model_path) |d| try writer.print("draft_model = \"{s}\"\n", .{d});
    }

    fn parseNextFloat(i: *usize, args: []const [:0]u8) !f32 {
        i.* += 1;
        if (i.* >= args.len) return ParseError.MissingArgument;
        return std.fmt.parseFloat(f32, args[i.*]) catch return ParseError.InvalidFloat;
    }

    fn parseNextInt(comptime T: type, i: *usize, args: []const [:0]u8) !T {
        i.* += 1;
        if (i.* >= args.len) return ParseError.MissingArgument;
        return std.fmt.parseInt(T, args[i.*], 10) catch return ParseError.InvalidInt;
    }

    fn getNextArg(i: *usize, args: []const [:0]u8) !?[]const u8 {
        i.* += 1;
        if (i.* >= args.len) return null;
        return args[i.*];
    }
};

test "parse config defaults" {
    // Tests need to use [:0]u8 match the signature
    // Since string literals are constant, we need to cast or copy
    const arg0 = try std.testing.allocator.dupeZ(u8, "exe");
    defer std.testing.allocator.free(arg0);
    const arg1 = try std.testing.allocator.dupeZ(u8, "model.gguf");
    defer std.testing.allocator.free(arg1);

    var args = [_][:0]u8{ arg0, arg1 };
    const cfg = try Config.parse(std.testing.allocator, &args);

    try std.testing.expectEqualStrings("model.gguf", cfg.model_path);
    try std.testing.expectEqual(@as(f32, 0.8), cfg.temp);
    try std.testing.expectEqual(@as(u32, 4096), cfg.n_ctx);
    try std.testing.expect(cfg.server_mode == false);
}

test "toml file layering" {
    const toml_text = "[model]\nn_ctx = 8192\n[sampling]\ntemp = 0.5\n";
    var cfg = Config{};
    try cfg.applyToml(std.testing.allocator, toml_text);
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 8192), cfg.n_ctx);
    try std.testing.expectEqual(@as(f32, 0.5), cfg.temp);
}

test "n_ctx auto sentinel" {
    var cfg = Config{};
    try cfg.applyToml(std.testing.allocator, "[model]\nn_ctx = \"auto\"\n");
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 0), cfg.n_ctx);
}

test "max_concurrent toml" {
    var cfg = Config{};
    try cfg.applyToml(std.testing.allocator, "[serve]\nmax_concurrent = 8\n");
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 8), cfg.max_concurrent);
}

test "prefix_cache toml" {
    var cfg = Config{};
    try cfg.applyToml(std.testing.allocator, "[serve]\nprefix_cache = true\n");
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expect(cfg.prefix_cache == true);
}

test "official residency toml" {
    var cfg = Config{};
    try cfg.applyToml(std.testing.allocator, "[residency]\nenabled = true\nbackend = \"ggml\"\nweight_budget_mib = 4\nstate_budget_mib = 512\n");
    defer cfg.deinit(std.testing.allocator);
    cfg.n_gpu_layers = 0;
    try cfg.validateResidency();
    try std.testing.expect(cfg.residency_enabled);
    try std.testing.expectEqualStrings("ggml", cfg.residency_backend);
    try std.testing.expectEqual(@as(usize, 4), cfg.residency_weight_budget_mib);
    try std.testing.expectEqual(@as(?usize, 512), cfg.residency_state_budget_mib);
}

test "official residency preflight rejects incompatible config" {
    var cfg = Config{ .residency_enabled = true, .n_gpu_layers = 0 };
    cfg.residency_weight_budget_mib = 0;
    try std.testing.expectError(Config.ParseError.ResidencyBudgetRequired, cfg.validateResidency());
    cfg.residency_weight_budget_mib = 4;
    cfg.n_gpu_layers = 1;
    try std.testing.expectError(Config.ParseError.ResidencyRequiresCpu, cfg.validateResidency());
    cfg.n_gpu_layers = 0;
    cfg.residency_backend = "other";
    try std.testing.expectError(Config.ParseError.InvalidResidencyBackend, cfg.validateResidency());
    cfg.residency_backend = "ggml";
    cfg.n_gpu_layers = 999;
    try cfg.validateResidency();
}
