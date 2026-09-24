#ifndef MLZ_RESIDENCY_MMAP_H
#define MLZ_RESIDENCY_MMAP_H

#include <stddef.h>
#include <stdint.h>

typedef struct mlz_backing_handle mlz_backing_handle;

typedef struct mlz_mapped_region {
    void *base;
    const unsigned char *data;
    size_t mapped_len;
} mlz_mapped_region;

mlz_backing_handle *mlz_backing_open(const char *path, uint64_t *size_out);
void mlz_backing_close(mlz_backing_handle *handle);
/* Alignment required for mapped file offsets, or zero when unavailable. On
 * Windows this is the allocation granularity, not the native page size. */
size_t mlz_backing_granularity(void);

/* Returns the native virtual-memory page size, or zero when it cannot be
 * queried. This may be smaller than mlz_backing_granularity() on Windows. */
size_t mlz_system_page_size(void);

int mlz_backing_map(mlz_backing_handle *handle, uint64_t offset, size_t len,
                    mlz_mapped_region *out);
void mlz_backing_unmap(mlz_mapped_region *region);

/* Synchronously reads one byte from every native OS page intersecting the
 * non-empty logical range [region->data, region->data + len), and also reads
 * the range's final byte. The volatile reads cannot be optimized away.
 *
 * The region must describe a live mapping and the requested logical range
 * must fit entirely between region->base and the end of region->mapped_len.
 * Returns 0 after all reads complete, or -1 for an invalid region/range or if
 * the native page size is unavailable. This faults pages in but does not lock
 * them in memory; passing stale or otherwise inaccessible mapping data has
 * undefined behavior. */
int mlz_mapped_region_prefault(const mlz_mapped_region *region, size_t len);

/* Process RSS helpers used by the real-GGUF validation tool. A zero return
 * means the metric is unavailable on the current platform. */
uint64_t mlz_process_current_rss(void);
uint64_t mlz_process_peak_rss(void);

#endif
