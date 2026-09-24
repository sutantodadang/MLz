const std = @import("std");
const context = @import("context.zig");
const Context = context.Context;

/// CUDA support - detect paths from environment variables and add CUDA
/// sources/include/library paths to the `ggml` static library. Sets
/// `ctx.cuda_so_output` (non-null on Linux; see `compileCudaSources`).
pub fn addToGgml(ctx: *Context, ggml_lib: *std.Build.Step.Compile) void {
    const b = ctx.b;
    const target = ctx.target;
    const llama_cpp_dep = ctx.llama_cpp_dep;

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

    // On Linux, add CUDA stubs path for libcuda.so driver API stub.
    // The CUDA toolkit ships stub libraries for CI/build environments
    // that don't have a physical GPU or NVIDIA driver installed.
    if (target.result.os.tag != .windows) {
        const cuda_stubs_path = b.pathJoin(&.{ cuda_root, "lib64", "stubs" });
        ggml_lib.addLibraryPath(.{ .cwd_relative = cuda_stubs_path });
    }

    // NOTE: Do NOT linkSystemLibrary for CUDA on ggml_lib (static archive).
    // LLD will warn/error about .so members in .a files.  The final exe
    // links CUDA libraries directly (see below).  On Linux, libstdc++ is
    // handled by the CUDA shared library (see compileCudaSources).

    // Compile CUDA sources with nvcc.  On Linux, this returns a LazyPath
    // to libggml-cuda.so which must be installed alongside the executable.
    ctx.cuda_so_output = compileCudaSources(b, ggml_lib, llama_cpp_dep, cuda_root, ggml_cuda_path_abs, target.result.os.tag);
}

pub fn addToLlama(ctx: *Context, llama_lib: *std.Build.Step.Compile) void {
    const b = ctx.b;
    const target = ctx.target;
    const cuda_path = getCudaPath(b).?;
    const cuda_lib_path = switch (target.result.os.tag) {
        .windows => b.pathJoin(&.{ cuda_path, "lib", "x64" }),
        else => b.pathJoin(&.{ cuda_path, "lib64" }),
    };
    llama_lib.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
    // Add CUDA stubs path for CI/build environments without a GPU driver.
    // The CUDA toolkit ships stub libraries (libcuda.so) needed at link time.
    if (target.result.os.tag != .windows) {
        const llama_cuda_stubs = b.pathJoin(&.{ cuda_path, "lib64", "stubs" });
        llama_lib.addLibraryPath(.{ .cwd_relative = llama_cuda_stubs });
    }
    // NOTE: Do NOT linkSystemLibrary for CUDA on llama_lib (static archive).
    // Same reasoning as ggml_lib — LLD warns about .so members in .a files.
    // The final exe links all CUDA libraries directly.
}

pub fn linkExecutable(ctx: *Context, exe: *std.Build.Step.Compile) void {
    const b = ctx.b;
    const target = ctx.target;
    const actual_target = ctx.actual_target;

    const cuda_path = getCudaPath(b).?;
    const cuda_lib_path = switch (target.result.os.tag) {
        .windows => b.pathJoin(&.{ cuda_path, "lib", "x64" }),
        else => b.pathJoin(&.{ cuda_path, "lib64" }),
    };
    exe.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
    // Add CUDA stubs path for CI/build environments without a GPU driver.
    if (target.result.os.tag != .windows) {
        const exe_cuda_stubs = b.pathJoin(&.{ cuda_path, "lib64", "stubs" });
        exe.addLibraryPath(.{ .cwd_relative = exe_cuda_stubs });
    }

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
        // On Linux, CUDA objects live in libggml-cuda.so (built by
        // compileCudaSources with g++ -shared -lstdc++).  Link the
        // .so directly to the exe so lld can resolve CUDA symbols.
        // Static archives (ggml_lib) cannot contain .so files.
        if (ctx.cuda_so_output) |cuda_so| {
            exe.addObjectFile(cuda_so);
        }
        // Set RPATH so the binary can find libggml-cuda.so at runtime
        // in the same directory as the executable.
        exe.root_module.addRPathSpecial("$ORIGIN");
    }
    exe.linkSystemLibrary("cublas");
    exe.linkSystemLibrary("cuda");
}

/// Install CUDA shared library alongside the executable (Linux only).
/// The .so encapsulates GNU libstdc++ dependencies and is found at
/// runtime via the $ORIGIN RPATH set on the executable.
pub fn installSharedLib(ctx: *Context) void {
    const b = ctx.b;
    if (ctx.cuda_so_output) |cuda_so| {
        const install_cuda_so = b.addInstallFile(cuda_so, "bin/libggml-cuda.so");
        b.getInstallStep().dependOn(&install_cuda_so.step);
    }
}

/// Add the CUDA `bin` directory to PATH for the `run` step so the executable
/// can find the CUDA runtime DLLs at runtime.
pub fn setRunEnv(ctx: *Context, run_cmd: *std.Build.Step.Run) void {
    const b = ctx.b;
    const target = ctx.target;
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

/// Get CUDA installation path from environment variable.
/// Supports CUDA_PATH (Windows default) and CUDA_HOME (Linux/macOS common).
pub fn getCudaPath(b: *std.Build) ?[]const u8 {
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

/// Compile CUDA source files (.cu) with nvcc.
/// Supports both Windows (MSVC host compiler) and Linux (g++) targets.
///
/// On Linux, CUDA objects are linked into a shared library (libggml-cuda.so)
/// with `-lstdc++` to isolate GNU libstdc++ from Zig's LLVM libc++.  The
/// returned `LazyPath` (non-null on Linux) must be installed alongside the
/// executable so the dynamic linker can find it at runtime.
///
/// On Windows, objects are added directly to `ggml_lib` and `null` is returned.
fn compileCudaSources(
    b: *std.Build,
    ggml_lib: *std.Build.Step.Compile,
    llama_cpp_dep: *std.Build.Dependency,
    cuda_root: []const u8,
    ggml_cuda_path_abs: []const u8,
    target_os: std.Target.Os.Tag,
) ?std.Build.LazyPath {
    const nvcc_path = if (target_os == .windows)
        b.pathJoin(&.{ cuda_root, "bin", "nvcc.exe" })
    else
        b.pathJoin(&.{ cuda_root, "bin", "nvcc" });

    // MSVC host-compiler paths (Windows only)
    var msvc_base: []const u8 = "";
    var sdk_include: []const u8 = "";
    var cl_path_win: []const u8 = "";
    var cl_dir_win: []const u8 = "";
    var include_var: []const u8 = "";

    if (target_os == .windows) {
        msvc_base = b.graph.env_map.get("VCToolsInstallDir") orelse
            "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207";
        // Construct the full SDK include path from WindowsSdkDir + WindowsSDKVersion.
        // WindowsSdkDir is a base dir (e.g. "C:/Program Files (x86)/Windows Kits/10/"),
        // so we need to append "Include/<version>" to reach ucrt/shared/um headers.
        const sdk_dir = b.graph.env_map.get("WindowsSdkDir") orelse
            "C:/Program Files (x86)/Windows Kits/10";
        const sdk_ver = b.graph.env_map.get("WindowsSDKVersion") orelse
            "10.0.26100.0";
        // Strip trailing backslash from WindowsSDKVersion if present (VS sets it as "10.0.xxxxx.0\\")
        const sdk_ver_clean = if (sdk_ver.len > 0 and sdk_ver[sdk_ver.len - 1] == '\\')
            sdk_ver[0 .. sdk_ver.len - 1]
        else
            sdk_ver;
        sdk_include = b.pathJoin(&.{ sdk_dir, "Include", sdk_ver_clean });
        cl_path_win = b.pathJoin(&.{ msvc_base, "bin", "Hostx64", "x64", "cl.exe" });
        cl_dir_win = b.pathJoin(&.{ msvc_base, "bin", "Hostx64", "x64" });
        include_var = b.fmt(
            "{s}/include;{s}/ucrt;{s}/shared;{s}/um",
            .{ msvc_base, sdk_include, sdk_include, sdk_include },
        );
    }

    // Discover and compile all .cu files.
    // Use cwd().openDir() instead of openDirAbsolute() because the
    // dependency-resolved path may be relative (e.g. .zig-cache/p/…).
    // On POSIX, openat(AT_FDCWD, path) handles both absolute and relative paths.
    var cuda_dir = std.fs.cwd().openDir(ggml_cuda_path_abs, .{ .iterate = true }) catch |err| {
        std.debug.panic("failed to open ggml-cuda dir: {s}: {any}", .{ ggml_cuda_path_abs, err });
    };
    defer cuda_dir.close();

    var walker = cuda_dir.walk(b.allocator) catch @panic("oom walking ggml-cuda");
    defer walker.deinit();

    // llama.cpp include paths (computed once)
    const inc_ggml_include = llama_cpp_dep.path("ggml/include").getPath(b);
    const inc_ggml_src = llama_cpp_dep.path("ggml/src").getPath(b);

    // On Linux, collect CUDA objects to link into a shared library with
    // libstdc++.  This isolates GNU libstdc++ symbols from the main binary's
    // LLVM libc++ to avoid duplicate/undefined symbol conflicts.  On Windows,
    // CUDA objects link directly into ggml_lib (MSVC runtime has no
    // such conflict with Zig's C++ runtime).
    var linux_cuda_objs: std.ArrayList(std.Build.LazyPath) = .empty;

    while (true) {
        const entry_opt = walker.next() catch @panic("walk failed");
        if (entry_opt == null) break;
        const entry = entry_opt.?;
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".cu")) continue;

        const cc = b.addSystemCommand(&.{nvcc_path});

        if (target_os == .windows) {
            // Windows: use MSVC as host compiler
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
        } else {
            // Linux: use g++ as host compiler (nvcc default).
            // The resulting .o files reference GNU libstdc++ symbols.
            // We handle the libc++/libstdc++ conflict by linking these
            // objects into a shared library (see below) rather than
            // adding them directly to the static ggml_lib.
            cc.addArg("-c");
            cc.addArg("-O3");
            cc.addArg("-std=c++17");
            cc.addArg("--extended-lambda");
            cc.addArg("-Xcompiler");
            cc.addArg("-fPIC");
            cc.addArg("-Xcompiler");
            cc.addArg("-w");
            // Disable glibc fortification to avoid __fprintf_chk / __*_chk
            // references that Zig's bundled libc does not expose.
            cc.addArg("-Xcompiler");
            cc.addArg("-U_FORTIFY_SOURCE");
            cc.addArg("-Xcompiler");
            cc.addArg("-D_FORTIFY_SOURCE=0");
        }

        // Common flags
        cc.addArg("-DGGML_USE_CUDA");
        cc.addArg("-D_CRT_SECURE_NO_WARNINGS");
        cc.addArg("-DGGML_VERSION=100");
        cc.addArg("-DGGML_COMMIT=unknown");

        // Include paths
        cc.addArg("-I");
        cc.addArg(b.pathJoin(&.{ cuda_root, "include" }));
        cc.addArg("-I");
        cc.addArg(ggml_cuda_path_abs);
        cc.addArg("-I");
        cc.addArg(inc_ggml_include);
        cc.addArg("-I");
        cc.addArg(inc_ggml_src);

        const source_path = b.fmt("{s}/{s}", .{ ggml_cuda_path_abs, entry.path });
        cc.addArg(source_path);

        cc.addArg("-o");
        const obj_ext = if (target_os == .windows) ".obj" else ".o";
        const obj_name = b.fmt("{s}{s}", .{ entry.path, obj_ext });
        const obj = cc.addOutputFileArg(obj_name);

        if (target_os == .windows) {
            ggml_lib.addObjectFile(obj);
        } else {
            linux_cuda_objs.append(b.allocator, obj) catch @panic("oom");
        }
    }

    // On Linux: link CUDA objects into libggml-cuda.so with g++ to isolate
    // GNU libstdc++ from Zig's LLVM libc++.  The shared library encapsulates
    // all libstdc++ dependencies; the dynamic linker keeps them in a separate
    // linking scope from the main binary's libc++.
    if (target_os != .windows and linux_cuda_objs.items.len > 0) {
        const cuda_lib_path = b.pathJoin(&.{ cuda_root, "lib64" });
        const link_so = b.addSystemCommand(&.{ "g++", "-shared", "-o" });
        const cuda_so = link_so.addOutputFileArg("libggml-cuda.so");
        for (linux_cuda_objs.items) |obj| {
            link_so.addFileArg(obj);
        }
        link_so.addArgs(&.{
            "-L",       cuda_lib_path,
            "-lcudart", "-lcublas",
            "-lstdc++",
        });
        // Return the .so path — the caller adds it directly to the exe
        // (not to ggml_lib, because static archives cannot contain .so files).
        return cuda_so;
    }
    return null;
}
