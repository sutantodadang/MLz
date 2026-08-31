const std = @import("std");
const service_mod = @import("residency_service.zig");
const executor_mod = @import("residency_executor.zig");
const llama = @import("llama_cpp.zig");

fn silentLog(_: llama.c.ggml_log_level, _: [*c]const u8, _: ?*anyopaque) callconv(.c) void {}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len < 3 or args.len > 7) {
        std.debug.print(
            "usage: residency_service <model.gguf> <prompt> [budget-mib] [max-tokens] [state-cache-mib] [state-workspace-mib]\n",
            .{},
        );
        return error.InvalidArguments;
    }
    const budget_mib = if (args.len >= 4)
        try std.fmt.parseInt(usize, args[3], 10)
    else
        16;
    const max_tokens = if (args.len >= 5)
        try std.fmt.parseInt(usize, args[4], 10)
    else
        32;
    if (budget_mib == 0 or max_tokens == 0 or budget_mib > std.math.maxInt(usize) / (1024 * 1024)) {
        return error.InvalidArguments;
    }
    var state_budget: ?executor_mod.StateBudget = null;
    if (args.len >= 6) {
        const cache_mib = try std.fmt.parseInt(usize, args[5], 10);
        const workspace_mib = if (args.len >= 7)
            try std.fmt.parseInt(usize, args[6], 10)
        else
            256;
        state_budget = .{
            .cache_bytes = cache_mib * 1024 * 1024,
            .workspace_bytes = workspace_mib * 1024 * 1024,
        };
    }

    const path = args[1];
    const prompt = args[2];

    // Silence llama.cpp logging so the smoke-run summary is readable.
    llama.c.llama_log_set(silentLog, null);

    var service = try service_mod.ResidencyService.open(allocator, path, .{
        .budget_bytes = budget_mib * 1024 * 1024,
    });
    defer service.close();

    const tokenized = try service.tokenize(prompt, false);
    defer allocator.free(tokenized.tokens);

    const result = try service.complete(.{
        .budget_bytes = budget_mib * 1024 * 1024,
    }, .{
        .max_tokens = max_tokens,
        .prompt_tokens = tokenized.tokens,
        .state_budget = state_budget,
    });
    defer allocator.free(result.text);
    defer allocator.free(result.tokens);

    const out = std.fs.File.stdout().deprecatedWriter();
    try out.print(
        "bounded-residency completion: model={s}, budget={d} MiB\n" ++
            "prompt tokens={d}, generated={d}, elapsed={d:.2} ms\n" ++
            "weight-map={d:.2}/{d:.2} MiB peak/budget, scratch={d:.2} KiB, activations={d:.2} KiB, " ++
            "attention={d:.2} KiB, kv={d:.2} KiB\n" ++
            "faults={d}, hits={d}, evictions={d}, rss={d:.2} MiB\n" ++
            "--- completion ---\n{s}\n",
        .{
            path,
            budget_mib,
            result.prompt_tokens,
            result.tokens.len,
            result.elapsed_ms,
            @as(f64, @floatFromInt(result.peak_mapped_weight_bytes)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(result.weight_budget_bytes)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(result.dequant_scratch_bytes)) / 1024.0,
            @as(f64, @floatFromInt(result.activation_bytes)) / 1024.0,
            @as(f64, @floatFromInt(result.attention_workspace_bytes)) / 1024.0,
            @as(f64, @floatFromInt(result.kv_cache_bytes)) / 1024.0,
            result.faults,
            result.hits,
            result.evictions,
            @as(f64, @floatFromInt(result.rss_bytes orelse 0)) / (1024.0 * 1024.0),
            result.text,
        },
    );
}
