//! MLz: Machine Learning in Zig
//!
//! This library provides bindings to llama.cpp and utilities for building
//! inference servers and applications.

const std = @import("std");

pub const llama_cpp = @import("llama_cpp.zig");
pub const inference = @import("inference.zig");
pub const server = @import("server.zig");
pub const openai = @import("openai.zig");
pub const chat = @import("chat.zig");
pub const models = @import("models.zig");
pub const model_manager = @import("model_manager.zig");
pub const embeddings = @import("embeddings.zig");

test {
    // Run tests in all imported modules
    std.testing.refAllDecls(@This());
}
