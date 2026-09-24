//! MLz: Machine Learning in Zig
//!
//! This library provides bindings to llama.cpp and utilities for building
//! inference servers and applications.

const std = @import("std");

pub const llama_cpp = @import("llama/llama_cpp.zig");
pub const inference = @import("engine/inference.zig");
pub const engine = @import("engine/engine.zig");
pub const scheduler = @import("engine/scheduler.zig");
pub const chat = @import("engine/chat.zig");
pub const server = @import("server/server.zig");
pub const openai = @import("server/openai.zig");
pub const model_manager = @import("server/model_manager.zig");
pub const embeddings = @import("server/embeddings.zig");
pub const models = @import("app/models.zig");
pub const residency = @import("residency/manager.zig");
pub const gguf_index = @import("residency/gguf_index.zig");
pub const residency_ggml_bridge = @import("residency/ggml_bridge.zig");
pub const residency_memory_policy = @import("residency/memory_policy.zig");

test {
    // Run tests in all imported modules
    std.testing.refAllDecls(@This());
}
