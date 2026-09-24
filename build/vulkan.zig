const std = @import("std");
const context = @import("context.zig");
const Context = context.Context;

/// Adds the Vulkan backend to the `ggml` static library: generates SPIR-V
/// shaders from the vendored .comp sources, compiles ggml-vulkan.cpp, and
/// wires up platform-specific Vulkan SDK include/library paths.
pub fn addToGgml(ctx: *Context, ggml_lib: *std.Build.Step.Compile) void {
    const b = ctx.b;
    const target = ctx.target;
    const llama_cpp_dep = ctx.llama_cpp_dep;

    // ── Vulkan Shader Generation ──
    // The ggml-vulkan backend requires SPIR-V shaders embedded as C++ source/headers.
    // We compile the upstream vulkan-shaders-gen tool (host-native) and run it at
    // build time against every .comp shader to produce ggml-vulkan-shaders.{hpp,cpp}.

    // Step 1: Build the shader generator as a native host executable
    const shader_gen_exe = b.addExecutable(.{
        .name = "vulkan-shaders-gen",
        .root_module = b.createModule(.{
            .target = b.graph.host,
            .optimize = .ReleaseFast,
        }),
    });
    shader_gen_exe.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp"),
        .flags = &.{"-std=c++17"},
    });
    shader_gen_exe.linkLibCpp();
    shader_gen_exe.linkLibC();

    // Step 2: Enumerate all .comp shader source files from the dependency
    const shader_dir_path = llama_cpp_dep.path("ggml/src/ggml-vulkan/vulkan-shaders");
    const shader_dir_abs = shader_dir_path.getPath(b);
    var shader_src_dir = if (std.fs.path.isAbsolute(shader_dir_abs))
        std.fs.openDirAbsolute(shader_dir_abs, .{ .iterate = true }) catch |err| {
            std.debug.panic("failed to open Vulkan shader dir (absolute): {s}: {any}", .{ shader_dir_abs, err });
        }
    else
        std.fs.cwd().openDir(shader_dir_abs, .{ .iterate = true }) catch |err| {
            std.debug.panic("failed to open Vulkan shader dir (relative): {s}: {any}", .{ shader_dir_abs, err });
        };
    defer shader_src_dir.close();

    var comp_files: std.ArrayList([]const u8) = .empty;
    var dir_iter = shader_src_dir.iterate();
    while (dir_iter.next() catch @panic("iterate vulkan shader dir failed")) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".comp")) {
            comp_files.append(b.allocator, b.allocator.dupe(u8, entry.name) catch @panic("OOM")) catch @panic("OOM");
        }
    }

    // Step 3: For each .comp file, run the shader generator to compile
    //         GLSL → SPIR-V and embed data into a .cpp translation unit.
    //         The generated .cpp files #include the generated hpp header
    //         (ggml-vulkan-shaders.hpp) which itself #include <cstdint>.
    //         NOTE: Do NOT use "-include cstdint" here — Zig 0.15's
    //         cross-compile C compilation caching chokes on the
    //         -include flag with system headers, producing spurious
    //         CacheCheckFailed errors for every .comp.cpp file.
    var shader_data_cpp_flags: std.ArrayList([]const u8) = .empty;
    shader_data_cpp_flags.append(b.allocator, "-std=c++17") catch @panic("OOM");

    for (comp_files.items) |comp_file| {
        const rel_path = b.fmt("ggml/src/ggml-vulkan/vulkan-shaders/{s}", .{comp_file});
        const run_gen = b.addRunArtifact(shader_gen_exe);
        run_gen.addArg("--glslc");
        run_gen.addArg("glslc");
        run_gen.addArg("--source");
        run_gen.addFileArg(llama_cpp_dep.path(rel_path));
        run_gen.addArg("--output-dir");
        _ = run_gen.addOutputDirectoryArg(b.fmt("vk-spirv-{s}", .{comp_file}));
        // The tool uses basename(target_hpp) to emit #include "..." in
        // each generated .cpp.  Without this, the .cpp starts with
        // #include "" which is a compile error.
        run_gen.addArg("--target-hpp");
        run_gen.addArg("ggml-vulkan-shaders.hpp");
        run_gen.addArg("--target-cpp");
        const gen_cpp = run_gen.addOutputFileArg(b.fmt("{s}.cpp", .{comp_file}));
        ggml_lib.addCSourceFile(.{
            .file = gen_cpp,
            .flags = shader_data_cpp_flags.items,
        });
    }

    // Step 4: Run the shader generator once without --source to produce
    //         the header file with extern declarations for all shaders.
    const run_gen_hpp = b.addRunArtifact(shader_gen_exe);
    run_gen_hpp.addArg("--output-dir");
    _ = run_gen_hpp.addOutputDirectoryArg("vk-spirv-hpp");
    run_gen_hpp.addArg("--target-hpp");
    const generated_hpp = run_gen_hpp.addOutputFileArg("ggml-vulkan-shaders.hpp");

    // Add generated header directory so #include "ggml-vulkan-shaders.hpp" resolves
    ggml_lib.addIncludePath(generated_hpp.dirname());

    // ── Vulkan Backend Source ──
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-vulkan/ggml-vulkan.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addIncludePath(llama_cpp_dep.path("ggml/src/ggml-vulkan"));

    // Platform-specific Vulkan SDK handling — include paths and library
    // search paths for compilation.  Do NOT linkSystemLibrary here because
    // ggml_lib is a static archive and LLD will warn about .so members in
    // the .a file.  The final exe links vulkan directly (see below).
    switch (target.result.os.tag) {
        .windows => {
            if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                const lib_path = b.pathJoin(&.{ sdk_path, "Lib" });
                ggml_lib.addLibraryPath(.{ .cwd_relative = lib_path });
                // Vulkan headers are at <VULKAN_SDK>/Include (not in default search paths)
                const inc_path = b.pathJoin(&.{ sdk_path, "Include" });
                ggml_lib.addSystemIncludePath(.{ .cwd_relative = inc_path });
            } else {
                std.log.warn("VULKAN_SDK environment variable not set. Vulkan build may fail.", .{});
            }
        },
        .linux => {
            // Use VULKAN_SDK if available, otherwise try system paths
            if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                const lib_path = b.pathJoin(&.{ sdk_path, "lib" });
                ggml_lib.addLibraryPath(.{ .cwd_relative = lib_path });
                const inc_path = b.pathJoin(&.{ sdk_path, "include" });
                ggml_lib.addSystemIncludePath(.{ .cwd_relative = inc_path });
            } else {
                // Fallback: add standard multiarch library path for cross-compilation
                // Zig's cross-compile linker doesn't search /usr/lib/<triple> by default.
                // Only add the path matching the target arch to avoid FileNotFound warnings.
                const linux_multiarch_dir: []const u8 = switch (target.result.cpu.arch) {
                    .aarch64 => "/usr/lib/aarch64-linux-gnu",
                    else => "/usr/lib/x86_64-linux-gnu",
                };
                ggml_lib.addLibraryPath(.{ .cwd_relative = linux_multiarch_dir });
                ggml_lib.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });
            }
        },
        .macos => {
            // macOS uses MoltenVK via Vulkan SDK
            if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                const lib_path = b.pathJoin(&.{ sdk_path, "lib" });
                ggml_lib.addLibraryPath(.{ .cwd_relative = lib_path });
                const inc_path = b.pathJoin(&.{ sdk_path, "include" });
                ggml_lib.addSystemIncludePath(.{ .cwd_relative = inc_path });
            }
        },
        else => {},
    }
}

/// Zig 0.15 does not propagate addLibraryPath from static archives linked
/// via linkLibrary.  Duplicate the platform-specific library paths so that
/// llama can resolve ggml's transitive system library deps (e.g. Vulkan).
pub fn addToLlama(ctx: *Context, llama_lib: *std.Build.Step.Compile) void {
    const b = ctx.b;
    const target = ctx.target;
    switch (target.result.os.tag) {
        .linux => {
            if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                llama_lib.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "lib" }) });
            } else {
                const llama_multiarch_dir: []const u8 = switch (target.result.cpu.arch) {
                    .aarch64 => "/usr/lib/aarch64-linux-gnu",
                    else => "/usr/lib/x86_64-linux-gnu",
                };
                llama_lib.addLibraryPath(.{ .cwd_relative = llama_multiarch_dir });
            }
        },
        .windows => {
            if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                llama_lib.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "Lib" }) });
            }
        },
        .macos => {
            if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                llama_lib.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "lib" }) });
            }
        },
        else => {},
    }
}

/// Vulkan: link the system library on the executable only.  Do NOT put
/// linkSystemLibrary on static archives (ggml_lib/llama_lib) because LLD
/// will try to include the .so as an archive member and warn/error.
/// Zig 0.15 also does not propagate addLibraryPath from linkLibrary'd
/// static archives, so we duplicate the search paths here.
pub fn linkExecutable(ctx: *Context, exe: *std.Build.Step.Compile) void {
    const b = ctx.b;
    const target = ctx.target;
    switch (target.result.os.tag) {
        .linux => {
            if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                exe.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "lib" }) });
            } else {
                const exe_multiarch_dir: []const u8 = switch (target.result.cpu.arch) {
                    .aarch64 => "/usr/lib/aarch64-linux-gnu",
                    else => "/usr/lib/x86_64-linux-gnu",
                };
                exe.addLibraryPath(.{ .cwd_relative = exe_multiarch_dir });
            }
            exe.linkSystemLibrary("vulkan");
        },
        .windows => {
            if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                exe.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "Lib" }) });
            }
            exe.linkSystemLibrary("vulkan-1");
        },
        .macos => {
            if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                exe.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "lib" }) });
            }
            exe.linkSystemLibrary("vulkan");
        },
        else => {
            exe.linkSystemLibrary("vulkan");
        },
    }
}
