const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const use_cuda = b.option(bool, "cuda", "Use CUDA for GPU acceleration") orelse false;
    const target = b.standardTargetOptions(.{});
    var actual_target = target;
    if (use_cuda) {
        actual_target.query.abi = .msvc;
        actual_target.query.cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 };
    }
    const optimize = b.standardOptimizeOption(.{});

    const use_vulkan = b.option(bool, "vulkan", "Use Vulkan for GPU acceleration") orelse false;

    // ggml's C sources intentionally do pointer arithmetic on null pointers
    // (e.g. for size calculations). In Debug this can trap under Zig/clang's
    // runtime checks/sanitizers, so compile ggml optimized even when the rest
    // of the project is Debug.
    const ggml_optimize: std.builtin.OptimizeMode = switch (optimize) {
        .Debug => .ReleaseFast,
        else => optimize,
    };
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    const mod = b.addModule("MLz", .{
        // The root source file is the "entry point" of this module. Users of
        // this module will only be able to access public declarations contained
        // in this file, which means that if you have declarations that you
        // intend to expose to consumers that were defined in other files part
        // of this module, you will have to make sure to re-export them from
        // the root file.
        .root_source_file = b.path("src/root.zig"),
        // Later on we'll use this module as the root module of a test executable
        // which requires us to specify a target.
        .target = actual_target,
    });

    // Here we define an executable. An executable needs to have a root module
    // which needs to expose a `main` function. While we could add a main function
    // to the module defined above, it's sometimes preferable to split business
    // logic and the CLI into two separate modules.
    //
    // If your goal is to create a Zig library for others to use, consider if
    // it might benefit from also exposing a CLI tool. A parser library for a
    // data serialization format could also bundle a CLI syntax checker, for example.
    //
    // If instead your goal is to create an executable, consider if users might
    // be interested in also being able to embed the core functionality of your
    // program in their own executable in order to avoid the overhead involved in
    // subprocessing your CLI tool.
    //
    // If neither case applies to you, feel free to delete the declaration you
    // don't need and to put everything under a single module.
    const llama_cpp_dep = b.dependency("llama_cpp", .{});

    var c_flags: std.ArrayList([]const u8) = .empty;
    var cpp_flags: std.ArrayList([]const u8) = .empty;

    c_flags.append(b.allocator, "-std=c11") catch @panic("OOM");
    c_flags.append(b.allocator, "-D_CRT_SECURE_NO_WARNINGS") catch @panic("OOM");
    c_flags.append(b.allocator, "-DGGML_VERSION=\"100\"") catch @panic("OOM");
    c_flags.append(b.allocator, "-DGGML_COMMIT=\"unknown\"") catch @panic("OOM");
    c_flags.append(b.allocator, "-mno-avx512f") catch @panic("OOM");

    cpp_flags.append(b.allocator, "-std=c++17") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-D_CRT_SECURE_NO_WARNINGS") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-DGGML_VERSION=\"100\"") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-DGGML_COMMIT=\"unknown\"") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-mno-avx512f") catch @panic("OOM");

    c_flags.append(b.allocator, "-DGGML_USE_CPU") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-DGGML_USE_CPU") catch @panic("OOM");

    if (use_vulkan) {
        c_flags.append(b.allocator, "-DGGML_USE_VULKAN") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-DGGML_USE_VULKAN") catch @panic("OOM");
    } else if (use_cuda) {
        c_flags.append(b.allocator, "-DGGML_USE_CUDA") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-DGGML_USE_CUDA") catch @panic("OOM");
    }

    const ggml_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "ggml",
        .root_module = b.createModule(.{
            .target = actual_target,
            .optimize = ggml_optimize,
        }),
    });

    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml.c"),
        .flags = c_flags.items,
    });
    // ggml-base (C++)
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-opt.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-quants.c"),
        .flags = c_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-alloc.c"),
        .flags = c_flags.items,
    });
    // GGUF container helpers (required by llama.cpp model loader)
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/gguf.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-backend.cpp"),
        .flags = cpp_flags.items,
    });
    // ggml (registry)
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-backend-reg.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-threading.cpp"),
        .flags = cpp_flags.items,
    });

    // CPU backend (linked statically into ggml when GGML_BACKEND_DL is off)
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/ggml-cpu.c"),
        .flags = c_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/quants.c"),
        .flags = c_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/ggml-cpu.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/repack.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/hbm.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/traits.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/amx/amx.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/amx/mmq.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/binary-ops.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/unary-ops.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/vec.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/ops.cpp"),
        .flags = cpp_flags.items,
    });
    // x86 feature detection + optimized repack/quants
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/arch/x86/repack.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/arch/x86/quants.c"),
        .flags = c_flags.items,
    });

    if (use_vulkan) {
        ggml_lib.addCSourceFile(.{
            .file = llama_cpp_dep.path("ggml/src/ggml-vulkan/ggml-vulkan.cpp"),
            .flags = cpp_flags.items,
        });
        ggml_lib.addIncludePath(llama_cpp_dep.path("ggml/src/ggml-vulkan"));
        if (target.result.os.tag == .windows) {
            if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                const lib_path = b.pathJoin(&.{ sdk_path, "Lib" });
                ggml_lib.addLibraryPath(.{ .cwd_relative = lib_path });
            }
        }
        ggml_lib.linkSystemLibrary("vulkan-1");
    } else if (use_cuda) {
        const cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1";
        ggml_lib.addIncludePath(.{ .cwd_relative = cuda_path ++ "/include" });
        const ggml_cuda_path_abs = llama_cpp_dep.path("ggml/src/ggml-cuda").getPath(b);
        ggml_lib.addIncludePath(.{ .cwd_relative = ggml_cuda_path_abs });
        ggml_lib.addLibraryPath(.{ .cwd_relative = cuda_path ++ "/lib/x64" });
        ggml_lib.linkSystemLibrary("cudart");
        ggml_lib.linkSystemLibrary("cublas");
        ggml_lib.linkSystemLibrary("cuda");

        const nvcc_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe";
        const msvc_base = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207";
        const sdk_base = "C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0";

        const cl_path_win = msvc_base ++ "/bin/Hostx64/x64/cl.exe";
        const cl_dir_win = msvc_base ++ "/bin/Hostx64/x64";

        const include_var = b.fmt(
            "{s}/include;{s}/ucrt;{s}/shared;{s}/um",
            .{ msvc_base, sdk_base, sdk_base, sdk_base },
        );

        // Discover and compile all .cu files
        var cuda_dir = std.fs.openDirAbsolute(ggml_cuda_path_abs, .{ .iterate = true }) catch |err| {
            std.debug.panic("failed to open ggml-cuda dir: {s}: {any}", .{ ggml_cuda_path_abs, err });
        };
        defer cuda_dir.close();

        var walker = cuda_dir.walk(b.allocator) catch @panic("oom walking ggml-cuda");
        defer walker.deinit();

        while (true) {
            const entry_opt = walker.next() catch @panic("walk failed");
            if (entry_opt == null) break;
            const entry = entry_opt.?;
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.path, ".cu")) continue;

            const cc = b.addSystemCommand(&.{nvcc_path});

            // Explicitly set PATH to include MSVC bin dir
            const current_path = std.process.getEnvVarOwned(b.allocator, "PATH") catch "";
            cc.setEnvironmentVariable("PATH", b.fmt("{s};{s}", .{ cl_dir_win, current_path }));
            cc.setEnvironmentVariable("INCLUDE", include_var);
            cc.addArg("-c");
            cc.addArg("-O3");
            cc.addArg("-std=c++17");
            cc.addArg("--extended-lambda");
            cc.addArg("--use-local-env");
            cc.addArg("-ccbin");
            cc.addArg(cl_path_win);

            cc.addArg("-Xcompiler");
            cc.addArg("/bigobj");
            cc.addArg("-Xcompiler");
            cc.addArg("/std:c++17");
            cc.addArg("-Xcompiler");
            cc.addArg("/w");

            cc.addArg("-DGGML_USE_CUDA");
            cc.addArg("-D_CRT_SECURE_NO_WARNINGS");
            cc.addArg("-DGGML_VERSION=100");
            cc.addArg("-DGGML_COMMIT=unknown");

            // CUDA paths
            cc.addArg("-I");
            cc.addArg("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\include");

            // llama.cpp paths
            const inc2 = llama_cpp_dep.path("ggml/include").getPath(b);
            const inc3 = llama_cpp_dep.path("ggml/src").getPath(b);

            cc.addArg("-I");
            cc.addArg(ggml_cuda_path_abs);
            cc.addArg("-I");
            cc.addArg(inc2);
            cc.addArg("-I");
            cc.addArg(inc3);

            const source_path = b.fmt("{s}/{s}", .{ ggml_cuda_path_abs, entry.path });
            cc.addArg(source_path);

            cc.addArg("-o");
            const obj_name = b.fmt("{s}.obj", .{entry.path});
            const obj = cc.addOutputFileArg(obj_name);

            ggml_lib.addObjectFile(obj);
        }

        ggml_lib.linkSystemLibrary("cudart_static");
        ggml_lib.linkSystemLibrary("cublas");
        ggml_lib.linkSystemLibrary("cuda");
        ggml_lib.addLibraryPath(.{ .cwd_relative = cuda_path ++ "/lib/x64" });
    }

    ggml_lib.addIncludePath(llama_cpp_dep.path("ggml/include"));
    ggml_lib.addIncludePath(llama_cpp_dep.path("ggml/src"));
    ggml_lib.addIncludePath(llama_cpp_dep.path("ggml/src/ggml-cpu"));
    ggml_lib.addIncludePath(b.path("src")); // To find ggml_shim.h
    ggml_lib.linkLibC();
    if (actual_target.query.abi != .msvc) {
        ggml_lib.linkLibCpp();
    }

    // Build llama.cpp as a static library so we can use the official llama.h C API.
    // We compile it optimized even in Debug for the same reason as ggml (avoids UB traps).
    const llama_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "llama",
        .root_module = b.createModule(.{
            .target = actual_target,
            .optimize = ggml_optimize,
        }),
    });

    llama_lib.addIncludePath(llama_cpp_dep.path("include"));
    llama_lib.addIncludePath(llama_cpp_dep.path("src"));
    llama_lib.addIncludePath(llama_cpp_dep.path("ggml/include"));
    llama_lib.addIncludePath(llama_cpp_dep.path("ggml/src"));
    llama_lib.addIncludePath(llama_cpp_dep.path("ggml/src/ggml-cpu"));

    if (use_cuda) {
        const cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1";
        llama_lib.addLibraryPath(.{ .cwd_relative = cuda_path ++ "/lib/x64" });
        llama_lib.linkSystemLibrary("cudart_static");
        llama_lib.linkSystemLibrary("cublas");
        llama_lib.linkSystemLibrary("cuda");
    }

    // Add all .cpp files from llama.cpp/src recursively (includes model registry).
    // This mirrors the upstream CMakeLists (simpler than keeping a huge list in sync).
    const llama_src_abs = llama_cpp_dep.path("src").getPath(b);
    var llama_dir = std.fs.openDirAbsolute(llama_src_abs, .{ .iterate = true }) catch |err| {
        std.debug.panic("failed to open llama.cpp src dir: {s}: {any}", .{ llama_src_abs, err });
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
            .flags = cpp_flags.items,
        });
    }

    llama_lib.linkLibC();
    if (actual_target.query.abi != .msvc) {
        llama_lib.linkLibCpp();
    }
    llama_lib.linkLibrary(ggml_lib);

    const exe = b.addExecutable(.{
        .name = "MLz",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = actual_target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "MLz", .module = mod },
            },
        }),
    });

    // Link GGML
    exe.linkLibrary(ggml_lib);
    exe.addIncludePath(llama_cpp_dep.path("ggml/include"));
    exe.root_module.addIncludePath(llama_cpp_dep.path("ggml/include"));

    // Link llama.cpp C API
    exe.linkLibrary(llama_lib);
    exe.addIncludePath(llama_cpp_dep.path("include"));
    exe.root_module.addIncludePath(llama_cpp_dep.path("include"));

    exe.linkLibC();

    if (use_cuda) {
        const cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1";
        exe.addLibraryPath(.{ .cwd_relative = cuda_path ++ "/lib/x64" });
        exe.linkSystemLibrary("cudart_static");
        exe.linkSystemLibrary("cublas");
        exe.linkSystemLibrary("cuda");

        // MSVC/SDK libs for linking
        const msvc_lib = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/lib/x64";
        const sdk_lib_base = "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0";
        exe.addLibraryPath(.{ .cwd_relative = msvc_lib });
        exe.addLibraryPath(.{ .cwd_relative = sdk_lib_base ++ "/ucrt/x64" });
        exe.addLibraryPath(.{ .cwd_relative = sdk_lib_base ++ "/um/x64" });

        if (actual_target.query.abi == .msvc) {
            exe.linkSystemLibrary("libcpmt");
        }
    }

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    if (use_cuda) {
        const cuda_bin = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/x64";
        const current_path = std.process.getEnvVarOwned(b.allocator, "PATH") catch "";
        run_cmd.setEnvironmentVariable("PATH", b.fmt("{s};{s}", .{ cuda_bin, current_path }));
    }
    run_step.dependOn(&run_cmd.step);

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const test_step = b.step("test", "Run tests");
    const mod_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = actual_target,
            .optimize = optimize,
        }),
    });
    test_step.dependOn(&b.addRunArtifact(mod_tests).step);
}
