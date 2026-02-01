const std = @import("std");

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

    // One-shot prompt mode
    prompt_mode: bool = false,
    user_prompt: ?[]const u8 = null,

    // Server Config
    server_mode: bool = false,
    server_host: []const u8 = "127.0.0.1",
    server_port: u16 = 8080,
    server_api_key: ?[]const u8 = null,

    pub const ParseError = error{
        MissingModelPath,
        InvalidFloat,
        InvalidInt,
        MissingArgument,
    };

    pub fn parse(allocator: std.mem.Allocator, args: []const [:0]u8) !Config {
        var cfg = Config{};
        var path_set = false;

        var i: usize = 1;
        while (i < args.len) : (i += 1) {
            const arg = args[i];

            if (std.mem.eql(u8, arg, "--temp")) {
                cfg.temp = try parseNextFloat(&i, args);
            } else if (std.mem.eql(u8, arg, "--prompt")) {
                i += 1;
                if (i < args.len) {
                    cfg.user_prompt = args[i];
                    cfg.prompt_mode = true;
                } else {
                    return ParseError.MissingArgument;
                }
            } else if (std.mem.eql(u8, arg, "--top-k")) {
                cfg.top_k = try parseNextInt(i32, &i, args);
            } else if (std.mem.eql(u8, arg, "--top-p")) {
                cfg.top_p = try parseNextFloat(&i, args);
            } else if (std.mem.eql(u8, arg, "--min-p")) {
                cfg.min_p = try parseNextFloat(&i, args);
            } else if (std.mem.eql(u8, arg, "--seed")) {
                cfg.seed = try parseNextInt(u32, &i, args);
            } else if (std.mem.eql(u8, arg, "--ctx")) {
                cfg.n_ctx = try parseNextInt(u32, &i, args);
            } else if (std.mem.eql(u8, arg, "--ngl")) {
                cfg.n_gpu_layers = try parseNextInt(i32, &i, args);
            } else if (std.mem.eql(u8, arg, "--threads")) {
                cfg.threads = try parseNextInt(i32, &i, args);
            } else if (std.mem.eql(u8, arg, "--stream")) {
                i += 1;
                if (i < args.len) {
                    const val = args[i];
                    cfg.stream = std.mem.eql(u8, val, "true") or std.mem.eql(u8, val, "1");
                }
            } else if (std.mem.eql(u8, arg, "--system")) {
                cfg.system_prompt = try getNextArg(&i, args);
            } else if (std.mem.eql(u8, arg, "--save-chat")) {
                cfg.save_chat_path = try getNextArg(&i, args);
            } else if (std.mem.eql(u8, arg, "--load-chat")) {
                cfg.load_chat_path = try getNextArg(&i, args);
            } else if (std.mem.eql(u8, arg, "--grammar")) {
                cfg.grammar_path = try getNextArg(&i, args);
            } else if (std.mem.eql(u8, arg, "--grammar-root")) {
                cfg.grammar_root = try getNextArg(&i, args) orelse "root";
            } else if (std.mem.eql(u8, arg, "--server")) {
                cfg.server_mode = true;
            } else if (std.mem.eql(u8, arg, "--host")) {
                cfg.server_host = try getNextArg(&i, args) orelse "127.0.0.1";
            } else if (std.mem.eql(u8, arg, "--port")) {
                cfg.server_port = try parseNextInt(u16, &i, args);
            } else if (std.mem.eql(u8, arg, "--api-key")) {
                cfg.server_api_key = try getNextArg(&i, args);
            } else if (std.mem.startsWith(u8, arg, "--")) {
                std.log.warn("Unknown argument: {s}", .{arg});
            } else {
                if (!path_set) {
                    cfg.model_path = arg;
                    path_set = true;
                }
            }
        }

        if (!path_set) return ParseError.MissingModelPath;
        _ = allocator;
        return cfg;
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
