const std = @import("std");
const context = @import("context.zig");
const Context = context.Context;
const cuda = @import("cuda.zig");
const metal = @import("metal.zig");
const vulkan = @import("vulkan.zig");

/// Build llama.cpp as a static library so we can use the official llama.h C API.
/// We compile it optimized even in Debug for the same reason as ggml (avoids UB traps).
pub fn create(ctx: *Context, ggml_lib: *std.Build.Step.Compile) *std.Build.Step.Compile {
    const b = ctx.b;
    const llama_cpp_dep = ctx.llama_cpp_dep;

    const llama_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "llama",
        .root_module = b.createModule(.{
            .target = ctx.actual_target,
            .optimize = ctx.ggml_optimize,
        }),
    });

    llama_lib.addIncludePath(llama_cpp_dep.path("include"));
    llama_lib.addIncludePath(llama_cpp_dep.path("common"));
    llama_lib.addIncludePath(llama_cpp_dep.path("src"));
    llama_lib.addIncludePath(llama_cpp_dep.path("ggml/include"));
    llama_lib.addIncludePath(llama_cpp_dep.path("ggml/src"));
    llama_lib.addIncludePath(llama_cpp_dep.path("ggml/src/ggml-cpu"));

    if (ctx.options.use_cuda) {
        cuda.addToLlama(ctx, llama_lib);
    }

    if (ctx.options.use_metal) {
        metal.addToLlama(ctx, llama_lib);
    }

    // Add all .cpp files from llama.cpp/src recursively (includes model registry).
    // This mirrors the upstream CMakeLists (simpler than keeping a huge list in sync).
    const llama_src_abs = llama_cpp_dep.path("src").getPath(b);
    var llama_dir = if (std.fs.path.isAbsolute(llama_src_abs))
        std.fs.openDirAbsolute(llama_src_abs, .{ .iterate = true }) catch |err| {
            std.debug.panic("failed to open llama.cpp src dir (absolute): {s}: {any}", .{ llama_src_abs, err });
        }
    else
        std.fs.cwd().openDir(llama_src_abs, .{ .iterate = true }) catch |err| {
            std.debug.panic("failed to open llama.cpp src dir (relative): {s}: {any}", .{ llama_src_abs, err });
        };
    defer llama_dir.close();

    var walker = llama_dir.walk(b.allocator) catch @panic("oom walking llama.cpp src");
    defer walker.deinit();

    while (true) {
        const entry_opt = walker.next() catch @panic("walk failed");
        if (entry_opt == null) break;
        const entry = entry_opt.?;
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".cpp")) continue;

        const rel = b.allocator.alloc(u8, "src/".len + entry.path.len) catch @panic("oom");
        @memcpy(rel[0.."src/".len], "src/");
        @memcpy(rel["src/".len..], entry.path);
        llama_lib.addCSourceFile(.{
            .file = llama_cpp_dep.path(rel),
            .flags = ctx.cpp_flags.items,
        });
    }

    llama_lib.addIncludePath(llama_cpp_dep.path("vendor"));
    llama_lib.addIncludePath(b.path("src/llama"));
    llama_lib.addIncludePath(b.path("src/residency"));

    inline for (.{ "caps", "lexer", "parser", "runtime", "string", "value" }) |jinja_src| {
        llama_lib.addCSourceFile(.{
            .file = llama_cpp_dep.path("common/jinja/" ++ jinja_src ++ ".cpp"),
            .flags = ctx.cpp_flags.items,
        });
    }
    llama_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("common/unicode.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    llama_lib.addCSourceFile(.{
        .file = b.path("src/llama/jinja_shim.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    llama_lib.addCSourceFile(.{
        .file = b.path("src/llama/llama_memory_shim.cpp"),
        .flags = ctx.cpp_flags.items,
    });

    llama_lib.linkLibC();
    if (ctx.actual_target.query.abi != .msvc) {
        llama_lib.linkLibCpp();
    }
    llama_lib.linkLibrary(ggml_lib);

    // Zig 0.15 does not propagate addLibraryPath from static archives linked
    // via linkLibrary.  Duplicate the platform-specific library paths so that
    // llama can resolve ggml's transitive system library deps (e.g. Vulkan).
    if (ctx.options.use_vulkan) {
        vulkan.addToLlama(ctx, llama_lib);
    }

    return llama_lib;
}
