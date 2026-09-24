const std = @import("std");
const context = @import("context.zig");
const Context = context.Context;

/// Metal backend for Apple Silicon GPU acceleration. Adds Objective-C/C++
/// sources and links the required Apple frameworks into the `ggml` static
/// library. No-op unless the target is macOS/iOS.
pub fn addToGgml(ctx: *Context, ggml_lib: *std.Build.Step.Compile) void {
    const b = ctx.b;
    const llama_cpp_dep = ctx.llama_cpp_dep;

    // Only attempt to link frameworks if we are building FOR macOS
    if (ctx.actual_target.result.os.tag == .macos or ctx.actual_target.result.os.tag == .ios) {
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

pub fn addToLlama(ctx: *Context, llama_lib: *std.Build.Step.Compile) void {
    const b = ctx.b;
    if (ctx.actual_target.result.os.tag == .macos or ctx.actual_target.result.os.tag == .ios) {
        if (b.sysroot) |sdk_root| {
            llama_lib.addFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_root, "System", "Library", "Frameworks" }) });
            llama_lib.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sdk_root, "usr", "include" }) });
        }
    }
}

/// Link Metal frameworks to executable and install the Metal shader plus
/// the headers it needs for on-the-fly compilation at runtime.
pub fn linkExecutable(ctx: *Context, exe: *std.Build.Step.Compile) void {
    const b = ctx.b;
    const llama_cpp_dep = ctx.llama_cpp_dep;
    if (ctx.actual_target.result.os.tag == .macos or ctx.actual_target.result.os.tag == .ios) {
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

/// Tell llama.cpp where to find the Metal shader file at run time.
pub fn setRunEnv(ctx: *Context, run_cmd: *std.Build.Step.Run) void {
    const b = ctx.b;
    run_cmd.setEnvironmentVariable("GGML_METAL_PATH_RESOURCES", b.getInstallPath(.bin, ""));
}
