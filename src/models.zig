//! Local model registry + `mlz models` management.
//!
//! Registry lives under a per-user data dir:
//!   Windows: %LOCALAPPDATA%\mlz\models
//!   else:    $HOME/.mlz/models   (or $XDG_DATA_HOME/mlz/models if set)
//!
//! A "model" is just a .gguf file in that dir. `pull` downloads one (resumable),
//! `list`/`resolve`/`remove` operate on the dir. Resolution by bare name lets the
//! server/CLI accept `--model qwen2.5-0.5b` instead of a full path.

const std = @import("std");

pub const Entry = struct {
    name: []u8, // file name including .gguf
    size: u64,
};

/// Absolute registry dir path (created if missing). Caller frees.
pub fn registryDir(allocator: std.mem.Allocator) ![]u8 {
    const base = blk: {
        if (@import("builtin").os.tag == .windows) {
            if (std.process.getEnvVarOwned(allocator, "LOCALAPPDATA")) |v| break :blk v else |_| {}
        } else {
            if (std.process.getEnvVarOwned(allocator, "XDG_DATA_HOME")) |v| break :blk v else |_| {}
            if (std.process.getEnvVarOwned(allocator, "HOME")) |v| {
                defer allocator.free(v);
                break :blk try std.fs.path.join(allocator, &.{ v, ".local", "share" });
            } else |_| {}
        }
        break :blk try allocator.dupe(u8, ".");
    };
    defer allocator.free(base);

    const dir = try std.fs.path.join(allocator, &.{ base, "mlz", "models" });
    std.fs.cwd().makePath(dir) catch |e| switch (e) {
        error.PathAlreadyExists => {},
        else => return e,
    };
    return dir;
}

/// List .gguf files in `dir`. Caller frees each `.name` and the slice.
pub fn list(allocator: std.mem.Allocator, dir: []const u8) ![]Entry {
    var d = std.fs.cwd().openDir(dir, .{ .iterate = true }) catch |e| switch (e) {
        error.FileNotFound => return &.{},
        else => return e,
    };
    defer d.close();

    var out: std.ArrayList(Entry) = .empty;
    errdefer {
        for (out.items) |it| allocator.free(it.name);
        out.deinit(allocator);
    }

    var it = d.iterate();
    while (try it.next()) |ent| {
        if (ent.kind != .file) continue;
        if (!std.mem.endsWith(u8, ent.name, ".gguf")) continue;
        const st = d.statFile(ent.name) catch continue;
        try out.append(allocator, .{ .name = try allocator.dupe(u8, ent.name), .size = st.size });
    }
    return out.toOwnedSlice(allocator);
}

/// Resolve a bare name (with or without .gguf) to a full path, or null if absent.
/// Caller frees the returned path.
pub fn resolvePath(allocator: std.mem.Allocator, dir: []const u8, name: []const u8) !?[]u8 {
    const candidates = [_][]const u8{ name, "" };
    for (candidates, 0..) |_, i| {
        const fname = if (i == 0)
            try allocator.dupe(u8, name)
        else
            try std.fmt.allocPrint(allocator, "{s}.gguf", .{name});
        defer allocator.free(fname);
        const full = try std.fs.path.join(allocator, &.{ dir, fname });
        if (std.fs.cwd().access(full, .{})) |_| return full else |_| allocator.free(full);
    }
    return null;
}

pub fn remove(dir: []const u8, name: []const u8) !void {
    var d = try std.fs.cwd().openDir(dir, .{});
    defer d.close();
    d.deleteFile(name) catch |e| switch (e) {
        error.FileNotFound => {
            // try with .gguf suffix
            var buf: [512]u8 = undefined;
            const alt = std.fmt.bufPrint(&buf, "{s}.gguf", .{name}) catch return e;
            try d.deleteFile(alt);
        },
        else => return e,
    };
}

/// Expand a pull source into a download URL + target filename.
/// Accepts:
///   - a full https:// URL                    -> filename = last path segment
///   - HuggingFace shorthand "owner/repo/file" -> resolve URL on main branch
/// Caller frees both returned slices.
pub fn resolveSource(allocator: std.mem.Allocator, src: []const u8) !struct { url: []u8, name: []u8 } {
    if (std.mem.startsWith(u8, src, "http://") or std.mem.startsWith(u8, src, "https://")) {
        const last = std.mem.lastIndexOfScalar(u8, src, '/') orelse return error.BadSource;
        const name = src[last + 1 ..];
        if (name.len == 0) return error.BadSource;
        return .{ .url = try allocator.dupe(u8, src), .name = try allocator.dupe(u8, name) };
    }
    // HuggingFace shorthand: owner/repo/path/to/file.gguf
    var slashes: usize = 0;
    for (src) |c| {
        if (c == '/') slashes += 1;
    }
    if (slashes < 2) return error.BadSource;
    const first = std.mem.indexOfScalar(u8, src, '/').?;
    const second = std.mem.indexOfScalarPos(u8, src, first + 1, '/').?;
    const owner_repo = src[0..second];
    const file_path = src[second + 1 ..];
    const last = std.mem.lastIndexOfScalar(u8, file_path, '/') orelse 0;
    const name = if (last == 0) file_path else file_path[last + 1 ..];
    const url = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}/resolve/main/{s}", .{ owner_repo, file_path });
    return .{ .url = url, .name = try allocator.dupe(u8, name) };
}

/// Download `url` into `dir/name`, resumable via a `.part` file + HTTP Range.
/// Prints simple progress to stderr. Returns the final path (caller frees).
pub fn pull(allocator: std.mem.Allocator, dir: []const u8, url: []const u8, name: []const u8) ![]u8 {
    const final_path = try std.fs.path.join(allocator, &.{ dir, name });
    errdefer allocator.free(final_path);
    if (std.fs.cwd().access(final_path, .{})) |_| {
        std.debug.print("Already present: {s}\n", .{final_path});
        return final_path;
    } else |_| {}

    const part_path = try std.fmt.allocPrint(allocator, "{s}.part", .{final_path});
    defer allocator.free(part_path);

    // Resume offset = existing .part size.
    var have: u64 = 0;
    if (std.fs.cwd().statFile(part_path)) |st| have = st.size else |_| {}

    var file = try std.fs.cwd().createFile(part_path, .{ .truncate = false, .read = true });
    var file_closed = false;
    defer if (!file_closed) file.close();
    try file.seekTo(have);

    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    var redirect_buf: [8 * 1024]u8 = undefined;
    var wbuf: [256 * 1024]u8 = undefined;
    var fw = file.writerStreaming(&wbuf);

    var range_buf: [64]u8 = undefined;
    var extra: []const std.http.Header = &.{};
    var range_hdr: [1]std.http.Header = undefined;
    if (have > 0) {
        const rv = try std.fmt.bufPrint(&range_buf, "bytes={d}-", .{have});
        range_hdr[0] = .{ .name = "range", .value = rv };
        extra = range_hdr[0..1];
        std.debug.print("Resuming {s} from {d} bytes\n", .{ name, have });
    } else {
        std.debug.print("Downloading {s}\n", .{name});
    }

    const res = try client.fetch(.{
        .location = .{ .url = url },
        .method = .GET,
        .response_writer = &fw.interface,
        .extra_headers = extra,
        .redirect_buffer = &redirect_buf,
    });
    try fw.interface.flush();

    // 200 means the server ignored our Range and re-sent the whole body from 0,
    // but we appended after `have` -> file is now corrupt. Restart clean.
    if (have > 0 and res.status == .ok) {
        try file.setEndPos(0);
        try file.seekTo(0);
        var fw2 = file.writerStreaming(&wbuf);
        const res2 = try client.fetch(.{
            .location = .{ .url = url },
            .method = .GET,
            .response_writer = &fw2.interface,
            .redirect_buffer = &redirect_buf,
        });
        try fw2.interface.flush();
        if (res2.status != .ok) return error.HttpFailed;
    } else if (res.status != .ok and res.status != .partial_content) {
        std.debug.print("HTTP {d} fetching {s}\n", .{ @intFromEnum(res.status), url });
        return error.HttpFailed;
    }

    file.close();
    file_closed = true;
    try std.fs.cwd().rename(part_path, final_path);
    std.debug.print("Saved {s}\n", .{final_path});
    return final_path;
}

// -----------------------------------------------------------------------------
// `mlz models ...` CLI entry point. Returns true if it handled the args.
// -----------------------------------------------------------------------------
pub fn runCli(allocator: std.mem.Allocator, args: []const []const u8) !bool {
    if (args.len < 2 or !std.mem.eql(u8, args[1], "models")) return false;

    const dir = try registryDir(allocator);
    defer allocator.free(dir);

    const sub = if (args.len >= 3) args[2] else "list";

    if (std.mem.eql(u8, sub, "dir")) {
        std.debug.print("{s}\n", .{dir});
    } else if (std.mem.eql(u8, sub, "list")) {
        const entries = try list(allocator, dir);
        defer {
            for (entries) |e| allocator.free(e.name);
            allocator.free(entries);
        }
        if (entries.len == 0) {
            std.debug.print("No models in {s}\n(use: mlz models pull <url|owner/repo/file.gguf>)\n", .{dir});
        } else {
            for (entries) |e| {
                const mib = @as(f64, @floatFromInt(e.size)) / (1024.0 * 1024.0);
                std.debug.print("{s:<48} {d:>9.1} MiB\n", .{ e.name, mib });
            }
        }
    } else if (std.mem.eql(u8, sub, "rm")) {
        if (args.len < 4) {
            std.debug.print("usage: mlz models rm <name>\n", .{});
            return true;
        }
        remove(dir, args[3]) catch |e| {
            std.debug.print("rm failed: {any}\n", .{e});
            return true;
        };
        std.debug.print("Removed {s}\n", .{args[3]});
    } else if (std.mem.eql(u8, sub, "pull")) {
        if (args.len < 4) {
            std.debug.print("usage: mlz models pull <url|owner/repo/file.gguf>\n", .{});
            return true;
        }
        const s = try resolveSource(allocator, args[3]);
        defer {
            allocator.free(s.url);
            allocator.free(s.name);
        }
        const out_name = if (args.len >= 5) args[4] else s.name;
        const path = pull(allocator, dir, s.url, out_name) catch |e| {
            std.debug.print("pull failed: {any}\n", .{e});
            return true;
        };
        allocator.free(path);
    } else {
        std.debug.print(
            \\usage: mlz models <command>
            \\  list                 list models in the registry
            \\  pull <src> [name]    download a .gguf (full URL or owner/repo/file.gguf)
            \\  rm <name>            delete a model
            \\  dir                  print the registry path
            \\
        , .{});
    }
    return true;
}

// -----------------------------------------------------------------------------
test "resolveSource: full url" {
    const a = std.testing.allocator;
    const s = try resolveSource(a, "https://example.com/path/qwen.gguf");
    defer {
        a.free(s.url);
        a.free(s.name);
    }
    try std.testing.expectEqualStrings("https://example.com/path/qwen.gguf", s.url);
    try std.testing.expectEqualStrings("qwen.gguf", s.name);
}

test "resolveSource: hf shorthand" {
    const a = std.testing.allocator;
    const s = try resolveSource(a, "Qwen/Qwen2.5-0.5B-GGUF/qwen2.5-0.5b-q8_0.gguf");
    defer {
        a.free(s.url);
        a.free(s.name);
    }
    try std.testing.expectEqualStrings("https://huggingface.co/Qwen/Qwen2.5-0.5B-GGUF/resolve/main/qwen2.5-0.5b-q8_0.gguf", s.url);
    try std.testing.expectEqualStrings("qwen2.5-0.5b-q8_0.gguf", s.name);
}

test "list + resolve + remove" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realpathAlloc(a, ".");
    defer a.free(dir);

    // create two fake gguf files + one non-gguf
    try tmp.dir.writeFile(.{ .sub_path = "a.gguf", .data = "xxxx" });
    try tmp.dir.writeFile(.{ .sub_path = "b.gguf", .data = "yy" });
    try tmp.dir.writeFile(.{ .sub_path = "note.txt", .data = "z" });

    const entries = try list(a, dir);
    defer {
        for (entries) |e| a.free(e.name);
        a.free(entries);
    }
    try std.testing.expectEqual(@as(usize, 2), entries.len);

    const p = try resolvePath(a, dir, "a");
    try std.testing.expect(p != null);
    a.free(p.?);

    const miss = try resolvePath(a, dir, "nope");
    try std.testing.expect(miss == null);

    try remove(dir, "a.gguf");
    const gone = try resolvePath(a, dir, "a");
    try std.testing.expect(gone == null);
}
