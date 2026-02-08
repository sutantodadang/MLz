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
    if (use_cuda and target.result.os.tag == .windows) {
        actual_target.query.abi = .msvc;
        // Don't enforce x86_64_v3 globally as it might cause illegal instruction errors on older CPUs
        // actual_target.query.cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 };
    }
    const optimize = b.standardOptimizeOption(.{});

    const use_vulkan = b.option(bool, "vulkan", "Use Vulkan for GPU acceleration") orelse false;

    const use_metal_default = target.result.os.tag == .macos or target.result.os.tag == .ios;
    const use_metal = b.option(bool, "metal", "Use Metal for GPU acceleration (macOS/iOS)") orelse use_metal_default;

    // Handle macOS SDK root detection
    if (actual_target.result.os.tag == .macos or actual_target.result.os.tag == .ios) {
        if (b.sysroot == null) {
            if (b.graph.env_map.get("SDKROOT")) |sdk_root| {
                b.sysroot = sdk_root;
            } else {
                // Try to detect via xcrun on macOS hosts
                const argv = &[_][]const u8{ "xcrun", "--show-sdk-path" };
                if (std.process.Child.run(.{ .allocator = b.allocator, .argv = argv })) |res| {
                    defer {
                        b.allocator.free(res.stdout);
                        b.allocator.free(res.stderr);
                    }
                    if (res.term == .Exited and res.term.Exited == 0) {
                        const trimmed = std.mem.trim(u8, res.stdout, " \n\r");
                        if (trimmed.len > 0) {
                            b.sysroot = b.allocator.dupe(u8, trimmed) catch @panic("OOM");
                        }
                    }
                } else |_| {}
            }
        }
    }

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

    cpp_flags.append(b.allocator, "-std=c++17") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-D_CRT_SECURE_NO_WARNINGS") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-DGGML_VERSION=\"100\"") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-DGGML_COMMIT=\"unknown\"") catch @panic("OOM");

    // x86_64-specific flags
    // AVX512 is enabled by default for better performance on supported CPUs
    // Use -Dno-avx512=true to disable for compatibility with older CPUs
    const no_avx512 = b.option(bool, "no-avx512", "Disable AVX512 for compatibility with older CPUs") orelse false;
    if (actual_target.result.cpu.arch == .x86_64 and no_avx512) {
        c_flags.append(b.allocator, "-mno-avx512f") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-mno-avx512f") catch @panic("OOM");
    }

    c_flags.append(b.allocator, "-DGGML_USE_CPU") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-DGGML_USE_CPU") catch @panic("OOM");

    if (actual_target.result.os.tag == .linux) {
        c_flags.append(b.allocator, "-D_GNU_SOURCE") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-D_GNU_SOURCE") catch @panic("OOM");
    }

    if (use_vulkan) {
        c_flags.append(b.allocator, "-DGGML_USE_VULKAN") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-DGGML_USE_VULKAN") catch @panic("OOM");
    } else if (use_cuda) {
        c_flags.append(b.allocator, "-DGGML_USE_CUDA") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-DGGML_USE_CUDA") catch @panic("OOM");
    }

    if (use_metal) {
        c_flags.append(b.allocator, "-DGGML_USE_METAL") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-DGGML_USE_METAL") catch @panic("OOM");
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
    // Architecture-specific CPU backend sources
    switch (actual_target.result.cpu.arch) {
        .x86_64 => {
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
        },
        .aarch64 => {
            // ARM NEON optimizations
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp"),
                .flags = cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-cpu/arch/arm/repack.cpp"),
                .flags = cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-cpu/arch/arm/quants.c"),
                .flags = c_flags.items,
            });
        },
        else => {
            // Fallback: no arch-specific optimizations
        },
    }

    if (use_vulkan) {
        ggml_lib.addCSourceFile(.{
            .file = llama_cpp_dep.path("ggml/src/ggml-vulkan/ggml-vulkan.cpp"),
            .flags = cpp_flags.items,
        });
        ggml_lib.addIncludePath(llama_cpp_dep.path("ggml/src/ggml-vulkan"));

        // Platform-specific Vulkan SDK handling
        switch (target.result.os.tag) {
            .windows => {
                if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                    const lib_path = b.pathJoin(&.{ sdk_path, "Lib" });
                    ggml_lib.addLibraryPath(.{ .cwd_relative = lib_path });
                } else {
                    std.log.warn("VULKAN_SDK environment variable not set. Vulkan build may fail.", .{});
                }
                ggml_lib.linkSystemLibrary("vulkan-1");
            },
            .linux => {
                ggml_lib.linkSystemLibrary("vulkan");
            },
            .macos => {
                // macOS uses MoltenVK via Vulkan SDK
                if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                    const lib_path = b.pathJoin(&.{ sdk_path, "lib" });
                    ggml_lib.addLibraryPath(.{ .cwd_relative = lib_path });
                }
                ggml_lib.linkSystemLibrary("vulkan");
            },
            else => {
                ggml_lib.linkSystemLibrary("vulkan");
            },
        }
    } else if (use_cuda) {
        // CUDA support - detect paths from environment variables
        const cuda_path = getCudaPath(b);
        if (cuda_path == null) {
            std.log.err("CUDA_PATH environment variable not set. Please set it to your CUDA installation directory.", .{});
            std.log.err("Example: export CUDA_PATH=/usr/local/cuda", .{});
            @panic("CUDA_PATH required for CUDA build");
        }
        const cuda_root = cuda_path.?;

        const cuda_include = b.pathJoin(&.{ cuda_root, "include" });
        const ggml_cuda_path_abs = llama_cpp_dep.path("ggml/src/ggml-cuda").getPath(b);

        ggml_lib.addIncludePath(.{ .cwd_relative = cuda_include });
        ggml_lib.addIncludePath(.{ .cwd_relative = ggml_cuda_path_abs });

        // Platform-specific CUDA library paths
        const cuda_lib_path = switch (target.result.os.tag) {
            .windows => b.pathJoin(&.{ cuda_root, "lib", "x64" }),
            else => b.pathJoin(&.{ cuda_root, "lib64" }),
        };
        ggml_lib.addLibraryPath(.{ .cwd_relative = cuda_lib_path });

        // Link CUDA libraries
        if (target.result.os.tag == .windows) {
            ggml_lib.linkSystemLibrary("cudart");
            ggml_lib.linkSystemLibrary("cublas");
            ggml_lib.linkSystemLibrary("cuda");
        } else {
            ggml_lib.linkSystemLibrary("cudart");
            ggml_lib.linkSystemLibrary("cublas");
            ggml_lib.linkSystemLibrary("cuda");
        }

        // Compile CUDA sources with nvcc (Windows-specific for now)
        if (target.result.os.tag == .windows) {
            compileCudaSources(b, ggml_lib, llama_cpp_dep, cuda_root, ggml_cuda_path_abs);
        }
    }

    // Metal backend for Apple Silicon GPU acceleration
    if (use_metal) {
        // Only attempt to link frameworks if we are building FOR macOS
        if (actual_target.result.os.tag == .macos or actual_target.result.os.tag == .ios) {
            // Get SDK root from sysroot (populated above)
            if (b.sysroot) |sdk_root| {
                ggml_lib.addFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_root, "System", "Library", "Frameworks" }) });
                // Also add system include path
                ggml_lib.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sdk_root, "usr", "include" }) });
            }

            // Metal backend include path
            ggml_lib.addIncludePath(llama_cpp_dep.path("ggml/src/ggml-metal"));

            // Objective-C flags for .m files
            var objc_flags: std.ArrayList([]const u8) = .empty;
            objc_flags.append(b.allocator, "-D_CRT_SECURE_NO_WARNINGS") catch @panic("OOM");
            objc_flags.append(b.allocator, "-DGGML_VERSION=\"100\"") catch @panic("OOM");
            objc_flags.append(b.allocator, "-DGGML_COMMIT=\"unknown\"") catch @panic("OOM");
            objc_flags.append(b.allocator, "-DGGML_USE_CPU") catch @panic("OOM");
            objc_flags.append(b.allocator, "-DGGML_USE_METAL") catch @panic("OOM");
            objc_flags.append(b.allocator, "-fno-objc-arc") catch @panic("OOM");

            // Metal C++ flags
            var metal_cpp_flags: std.ArrayList([]const u8) = .empty;
            metal_cpp_flags.append(b.allocator, "-std=c++17") catch @panic("OOM");
            metal_cpp_flags.append(b.allocator, "-D_CRT_SECURE_NO_WARNINGS") catch @panic("OOM");
            metal_cpp_flags.append(b.allocator, "-DGGML_VERSION=\"100\"") catch @panic("OOM");
            metal_cpp_flags.append(b.allocator, "-DGGML_COMMIT=\"unknown\"") catch @panic("OOM");
            metal_cpp_flags.append(b.allocator, "-DGGML_USE_CPU") catch @panic("OOM");
            metal_cpp_flags.append(b.allocator, "-DGGML_USE_METAL") catch @panic("OOM");

            // Objective-C sources
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-context.m"),
                .flags = objc_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-device.m"),
                .flags = objc_flags.items,
            });

            // C++ sources
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal.cpp"),
                .flags = metal_cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-common.cpp"),
                .flags = metal_cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-device.cpp"),
                .flags = metal_cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-ops.cpp"),
                .flags = metal_cpp_flags.items,
            });

            // Link Apple frameworks required for Metal
            ggml_lib.linkFramework("Metal");
            ggml_lib.linkFramework("Foundation");
            ggml_lib.linkFramework("MetalPerformanceShaders");
            ggml_lib.linkFramework("MetalPerformanceShadersGraph");
        }
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
        const cuda_path = getCudaPath(b).?;
        const cuda_lib_path = switch (target.result.os.tag) {
            .windows => b.pathJoin(&.{ cuda_path, "lib", "x64" }),
            else => b.pathJoin(&.{ cuda_path, "lib64" }),
        };
        llama_lib.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
        if (target.result.os.tag == .windows) {
            llama_lib.linkSystemLibrary("cudart_static");
        } else {
            llama_lib.linkSystemLibrary("cudart");
        }
        llama_lib.linkSystemLibrary("cublas");
        llama_lib.linkSystemLibrary("cuda");
    }

    if (use_metal) {
        if (actual_target.result.os.tag == .macos or actual_target.result.os.tag == .ios) {
            if (b.sysroot) |sdk_root| {
                llama_lib.addFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_root, "System", "Library", "Frameworks" }) });
                llama_lib.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sdk_root, "usr", "include" }) });
            }
        }
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
        const cuda_path = getCudaPath(b).?;
        const cuda_lib_path = switch (target.result.os.tag) {
            .windows => b.pathJoin(&.{ cuda_path, "lib", "x64" }),
            else => b.pathJoin(&.{ cuda_path, "lib64" }),
        };
        exe.addLibraryPath(.{ .cwd_relative = cuda_lib_path });

        if (target.result.os.tag == .windows) {
            exe.linkSystemLibrary("cudart_static");

            // MSVC/SDK libs for linking - detect from environment
            if (getMsvcLibPath(b)) |msvc_lib| {
                exe.addLibraryPath(.{ .cwd_relative = msvc_lib });
            }
            if (getWindowsSdkLibPath(b, "ucrt")) |ucrt_lib| {
                exe.addLibraryPath(.{ .cwd_relative = ucrt_lib });
            }
            if (getWindowsSdkLibPath(b, "um")) |um_lib| {
                exe.addLibraryPath(.{ .cwd_relative = um_lib });
            }

            if (actual_target.query.abi == .msvc) {
                exe.linkSystemLibrary("libcpmt");
            }
        } else {
            exe.linkSystemLibrary("cudart");
        }
        exe.linkSystemLibrary("cublas");
        exe.linkSystemLibrary("cuda");
    }

    // Link Metal frameworks to executable
    if (use_metal) {
        if (actual_target.result.os.tag == .macos or actual_target.result.os.tag == .ios) {
            // Get SDK root from sysroot
            if (b.sysroot) |sdk_root| {
                exe.addFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_root, "System", "Library", "Frameworks" }) });
            }

            exe.linkFramework("Metal");
            exe.linkFramework("Foundation");
            exe.linkFramework("MetalPerformanceShaders");
            exe.linkFramework("MetalPerformanceShadersGraph");

            // Install the Metal shader file so it can be found at runtime
            const install_metal = b.addInstallFile(llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal.metal"), "bin/ggml-metal.metal");
            b.getInstallStep().dependOn(&install_metal.step);

            // Headers required for Metal on-the-fly compilation
            const install_common_h = b.addInstallFile(llama_cpp_dep.path("ggml/src/ggml-common.h"), "bin/ggml-common.h");
            b.getInstallStep().dependOn(&install_common_h.step);

            const install_metal_common_h = b.addInstallFile(llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-common.h"), "bin/ggml-metal-common.h");
            b.getInstallStep().dependOn(&install_metal_common_h.step);

            const install_metal_impl_h = b.addInstallFile(llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-impl.h"), "bin/ggml-metal-impl.h");
            b.getInstallStep().dependOn(&install_metal_impl_h.step);

            const install_metal_device_h = b.addInstallFile(llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-device.h"), "bin/ggml-metal-device.h");
            b.getInstallStep().dependOn(&install_metal_device_h.step);

            const install_metal_ops_h = b.addInstallFile(llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-ops.h"), "bin/ggml-metal-ops.h");
            b.getInstallStep().dependOn(&install_metal_ops_h.step);
        }
    }

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);

    // Ensure the Metal shader is installed before running
    run_cmd.step.dependOn(b.getInstallStep());

    // Tell llama.cpp where to find the Metal shader file
    if (use_metal) {
        run_cmd.setEnvironmentVariable("GGML_METAL_PATH_RESOURCES", b.getInstallPath(.bin, ""));
    }

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    run_step.dependOn(&run_cmd.step);

    if (use_cuda) {
        if (getCudaPath(b)) |cuda_path| {
            const cuda_bin = switch (target.result.os.tag) {
                .windows => b.pathJoin(&.{ cuda_path, "bin" }),
                else => b.pathJoin(&.{ cuda_path, "bin" }),
            };
            const current_path = std.process.getEnvVarOwned(b.allocator, "PATH") catch "";
            const path_sep = if (target.result.os.tag == .windows) ";" else ":";
            run_cmd.setEnvironmentVariable("PATH", b.fmt("{s}{s}{s}", .{ cuda_bin, path_sep, current_path }));
        }
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

/// Get CUDA installation path from environment variable.
/// Supports CUDA_PATH (Windows default) and CUDA_HOME (Linux/macOS common).
fn getCudaPath(b: *std.Build) ?[]const u8 {
    // Try CUDA_PATH first (Windows default)
    if (b.graph.env_map.get("CUDA_PATH")) |path| {
        return path;
    }
    // Try CUDA_HOME (common on Linux/macOS)
    if (b.graph.env_map.get("CUDA_HOME")) |path| {
        return path;
    }
    // Try common default locations
    const default_paths = [_][]const u8{
        "/usr/local/cuda",
        "/opt/cuda",
    };
    for (default_paths) |path| {
        if (std.fs.accessAbsolute(path, .{})) |_| {
            return path;
        } else |_| {}
    }
    return null;
}

/// Get MSVC library path from environment or common locations.
fn getMsvcLibPath(b: *std.Build) ?[]const u8 {
    // Try VCToolsInstallDir environment variable
    if (b.graph.env_map.get("VCToolsInstallDir")) |vc_dir| {
        return b.pathJoin(&.{ vc_dir, "lib", "x64" });
    }

    // Try to detect from VSINSTALLDIR
    if (b.graph.env_map.get("VSINSTALLDIR")) |vs_dir| {
        // This is a simplified detection - in practice you'd need to find the version
        const vc_base = b.pathJoin(&.{ vs_dir, "VC", "Tools", "MSVC" });
        // For now, return null if we can't find it precisely
        _ = vc_base;
    }

    return null;
}

/// Get Windows SDK library path.
fn getWindowsSdkLibPath(b: *std.Build, lib_type: []const u8) ?[]const u8 {
    // Try WindowsSdkDir environment variable
    if (b.graph.env_map.get("WindowsSdkDir")) |sdk_dir| {
        if (b.graph.env_map.get("WindowsSDKVersion")) |sdk_ver| {
            return b.pathJoin(&.{ sdk_dir, "Lib", sdk_ver, lib_type, "x64" });
        }
    }
    return null;
}

/// Compile CUDA source files with nvcc (Windows).
fn compileCudaSources(
    b: *std.Build,
    ggml_lib: *std.Build.Step.Compile,
    llama_cpp_dep: *std.Build.Dependency,
    cuda_root: []const u8,
    ggml_cuda_path_abs: []const u8,
) void {
    const nvcc_path = b.pathJoin(&.{ cuda_root, "bin", "nvcc.exe" });

    // Get MSVC paths from environment or use defaults
    const msvc_base = b.graph.env_map.get("VCToolsInstallDir") orelse
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207";
    const sdk_include = b.graph.env_map.get("WindowsSdkDir") orelse
        "C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0";

    const cl_path_win = b.pathJoin(&.{ msvc_base, "bin", "Hostx64", "x64", "cl.exe" });
    const cl_dir_win = b.pathJoin(&.{ msvc_base, "bin", "Hostx64", "x64" });

    const include_var = b.fmt(
        "{s}/include;{s}/ucrt;{s}/shared;{s}/um",
        .{ msvc_base, sdk_include, sdk_include, sdk_include },
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

        // CUDA include path
        cc.addArg("-I");
        cc.addArg(b.pathJoin(&.{ cuda_root, "include" }));

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
}
