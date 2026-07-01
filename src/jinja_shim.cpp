#include <string>
#include <cstring>
#include <algorithm>

#include "jinja_shim.h"
#include "nlohmann/json.hpp"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

int32_t mlz_render_chat_template(const char* tmpl, const char* messages_json,
                                 int32_t add_generation_prompt,
                                 char* buf, int32_t buf_len) {
    try {
        nlohmann::ordered_json msgs = nlohmann::ordered_json::parse(messages_json ? messages_json : "[]");

        nlohmann::ordered_json vars = nlohmann::ordered_json::object();
        vars["messages"] = msgs;
        vars["add_generation_prompt"] = (add_generation_prompt != 0);
        vars["bos_token"] = "";
        vars["eos_token"] = "";

        std::string tmpl_s(tmpl ? tmpl : "");

        jinja::lexer lexer;
        auto lr = lexer.tokenize(tmpl_s);
        jinja::program ast = jinja::parse_from_tokens(lr);

        jinja::context ctx(tmpl_s);
        jinja::global_from_json(ctx, vars, /*mark_input*/ true);

        jinja::runtime rt(ctx);
        auto results = rt.execute(ast);
        auto parts = jinja::runtime::gather_string_parts(results);
        const std::string out = jinja::render_string_parts(parts);

        int32_t n = (int32_t)out.size();
        if (buf && buf_len > 0) {
            int32_t cpy = std::min(n, buf_len - 1);
            std::memcpy(buf, out.data(), (size_t)cpy);
            buf[cpy] = '\0';
        }
        return n;
    } catch (...) {
        return -1;
    }
}
