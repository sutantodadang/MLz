const std = @import("std");

const context = @import("build/context.zig");
const ggml = @import("build/ggml.zig");
const vulkan = @import("build/vulkan.zig");
const cuda = @import("build/cuda.zig");
const metal = @import("build/metal.zig");
const simd = @import("build/simd.zig");
const llama = @import("build/llama.zig");
const artifacts = @import("build/artifacts.zig");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    var ctx = context.init(b);

    // This creates a module, which represents a collection of source files
    // alongside some compilation options. Zig modules are the preferred way
    // of making Zig code available to consumers. addModule defines a module
    // that we intend to make available for importing to our consumers.
    const mod = b.addModule("MLz", .{
        .root_source_file = b.path("src/root.zig"),
        .target = ctx.actual_target,
        .link_libc = true,
    });
    mod.addIncludePath(b.path("src/llama"));
    mod.addIncludePath(b.path("src/residency"));
    mod.addIncludePath(ctx.llama_cpp_dep.path("include"));
    mod.addIncludePath(ctx.llama_cpp_dep.path("ggml/include"));
    mod.addIncludePath(ctx.llama_cpp_dep.path("ggml/src"));
    mod.addCSourceFile(.{
        .file = b.path("src/residency/residency_mmap.c"),
        .flags = &.{"-std=c11"},
    });

    const ggml_lib = ggml.create(&ctx);

    if (ctx.options.use_vulkan) {
        vulkan.addToGgml(&ctx, ggml_lib);
    } else if (ctx.options.use_cuda) {
        cuda.addToGgml(&ctx, ggml_lib);
    }

    // Metal backend for Apple Silicon GPU acceleration
    if (ctx.options.use_metal) {
        metal.addToGgml(&ctx, ggml_lib);
    }

    // Custom SIMD backend for high-performance matrix multiplication
    simd.addToGgml(&ctx, ggml_lib);

    ggml.finalize(&ctx, ggml_lib);

    const llama_lib = llama.create(&ctx, ggml_lib);

    artifacts.build(&ctx, ggml_lib, llama_lib, mod);
}
