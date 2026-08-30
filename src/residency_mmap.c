#include "residency_mmap.h"

#include <stdlib.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define PSAPI_VERSION 2
#include <windows.h>
#include <psapi.h>

typedef struct mlz_backing_handle {
    HANDLE file;
    HANDLE mapping;
    uint64_t size;
} mlz_backing_handle;

#else
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct mlz_backing_handle {
    int fd;
    uint64_t size;
} mlz_backing_handle;
#endif

mlz_backing_handle *mlz_backing_open(const char *path, uint64_t *size_out) {
    if (path == NULL || size_out == NULL) {
        return NULL;
    }

    mlz_backing_handle *handle = (mlz_backing_handle *) calloc(1, sizeof(*handle));
    if (handle == NULL) {
        return NULL;
    }

#if defined(_WIN32)
    handle->file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (handle->file == INVALID_HANDLE_VALUE) {
        free(handle);
        return NULL;
    }

    LARGE_INTEGER size;
    if (!GetFileSizeEx(handle->file, &size) || size.QuadPart < 0) {
        CloseHandle(handle->file);
        free(handle);
        return NULL;
    }
    handle->size = (uint64_t) size.QuadPart;

    /* Windows cannot create a file mapping for an empty file. */
    handle->mapping = handle->size == 0
        ? NULL
        : CreateFileMappingA(handle->file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (handle->size != 0 && handle->mapping == NULL) {
        CloseHandle(handle->file);
        free(handle);
        return NULL;
    }
#else
    handle->fd = open(path, O_RDONLY);
    if (handle->fd < 0) {
        free(handle);
        return NULL;
    }

    struct stat st;
    if (fstat(handle->fd, &st) != 0 || st.st_size < 0) {
        close(handle->fd);
        free(handle);
        return NULL;
    }
    handle->size = (uint64_t) st.st_size;
#endif

    *size_out = handle->size;
    return handle;
}

void mlz_backing_close(mlz_backing_handle *handle) {
    if (handle == NULL) {
        return;
    }
#if defined(_WIN32)
    if (handle->mapping != NULL) {
        CloseHandle(handle->mapping);
    }
    if (handle->file != INVALID_HANDLE_VALUE) {
        CloseHandle(handle->file);
    }
#else
    if (handle->fd >= 0) {
        close(handle->fd);
    }
#endif
    free(handle);
}

size_t mlz_backing_granularity(void) {
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return (size_t) info.dwAllocationGranularity;
#else
    return mlz_system_page_size();
#endif
}

size_t mlz_system_page_size(void) {
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return (size_t) info.dwPageSize;
#else
    long page_size = sysconf(_SC_PAGESIZE);
    return page_size > 0 ? (size_t) page_size : 0;
#endif
}

int mlz_backing_map(mlz_backing_handle *handle, uint64_t offset, size_t len,
                    mlz_mapped_region *out) {
    if (handle == NULL || out == NULL || len == 0 ||
        offset > handle->size || (uint64_t) len > handle->size - offset) {
        return -1;
    }

#if defined(_WIN32)
    const uint64_t granularity = (uint64_t) mlz_backing_granularity();
    if (granularity == 0) {
        return -1;
    }
    const uint64_t aligned = offset - (offset % granularity);
    const size_t delta = (size_t) (offset - aligned);
    if (len > SIZE_MAX - delta) {
        return -1;
    }
    const size_t mapped_len = delta + len;
    const DWORD high = (DWORD) (aligned >> 32);
    const DWORD low = (DWORD) aligned;
    void *base = MapViewOfFile(handle->mapping, FILE_MAP_READ, high, low, mapped_len);
    if (base == NULL) {
        return -1;
    }
#else
    const uint64_t page_size = (uint64_t) mlz_backing_granularity();
    if (page_size == 0) {
        return -1;
    }
    const uint64_t aligned = offset - (offset % page_size);
    const size_t delta = (size_t) (offset - aligned);
    if (len > SIZE_MAX - delta) {
        return -1;
    }
    const size_t mapped_len = delta + len;
    void *base = mmap(NULL, mapped_len, PROT_READ, MAP_PRIVATE, handle->fd,
                      (off_t) aligned);
    if (base == MAP_FAILED) {
        return -1;
    }
#endif

    out->base = base;
    out->data = (const unsigned char *) base + delta;
    out->mapped_len = mapped_len;
    return 0;
}

void mlz_backing_unmap(mlz_mapped_region *region) {
    if (region == NULL || region->base == NULL) {
        return;
    }
#if defined(_WIN32)
    UnmapViewOfFile(region->base);
#else
    munmap(region->base, region->mapped_len);
#endif
    region->base = NULL;
    region->data = NULL;
    region->mapped_len = 0;
}

int mlz_mapped_region_prefault(const mlz_mapped_region *region, size_t len) {
    if (region == NULL || region->base == NULL || region->data == NULL ||
        region->mapped_len == 0 || len == 0) {
        return -1;
    }

    const uintptr_t base_address = (uintptr_t) region->base;
    const uintptr_t data_address = (uintptr_t) region->data;
    if (data_address < base_address ||
        region->mapped_len > UINTPTR_MAX - base_address) {
        return -1;
    }

    const uintptr_t data_offset = data_address - base_address;
    if (data_offset > SIZE_MAX ||
        (size_t) data_offset > region->mapped_len ||
        len > region->mapped_len - (size_t) data_offset ||
        len - 1 > UINTPTR_MAX - data_address) {
        return -1;
    }

    const size_t page_size = mlz_system_page_size();
    if (page_size == 0) {
        return -1;
    }

    const volatile unsigned char *data =
        (const volatile unsigned char *) region->data;

    /* Touch the first page, then each subsequent page at its native virtual
     * address boundary. Reading the final byte is explicit even when it lies
     * on a page already touched above. */
    (void) data[0];
    size_t offset = page_size - (size_t) (data_address % (uintptr_t) page_size);
    while (offset < len) {
        (void) data[offset];
        if (page_size > SIZE_MAX - offset) {
            break;
        }
        offset += page_size;
    }
    (void) data[len - 1];
    return 0;
}

uint64_t mlz_process_current_rss(void) {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters;
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return 0;
    }
    return (uint64_t) counters.WorkingSetSize;
#elif defined(__linux__)
    FILE *file = fopen("/proc/self/statm", "r");
    if (file == NULL) {
        return 0;
    }
    unsigned long ignored_pages = 0;
    unsigned long resident_pages = 0;
    const int parsed = fscanf(file, "%lu %lu", &ignored_pages, &resident_pages);
    fclose(file);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (parsed != 2 || page_size <= 0) {
        return 0;
    }
    return (uint64_t) resident_pages * (uint64_t) page_size;
#elif defined(__APPLE__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
    return (uint64_t) usage.ru_maxrss;
#else
    return 0;
#endif
}

uint64_t mlz_process_peak_rss(void) {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters;
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return 0;
    }
    return (uint64_t) counters.PeakWorkingSetSize;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
#if defined(__APPLE__)
    return (uint64_t) usage.ru_maxrss;
#else
    return (uint64_t) usage.ru_maxrss * 1024u;
#endif
#endif
}
