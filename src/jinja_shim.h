#ifndef MLZ_JINJA_SHIM_H
#define MLZ_JINJA_SHIM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Render a jinja chat template. messages_json = JSON array of {"role","content"}.
   Returns rendered length (>=0); writes up to buf_len-1 bytes + NUL when buf!=NULL.
   Returns -1 on any error/exception. Call with buf=NULL,buf_len=0 to size. */
int32_t mlz_render_chat_template(const char* tmpl, const char* messages_json,
                                 int32_t add_generation_prompt,
                                 char* buf, int32_t buf_len);
#ifdef __cplusplus
}
#endif
#endif
