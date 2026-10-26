#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ucontext.h>

static void crash(int signal_number, siginfo_t *info, void *context) {
    dprintf(2, "NATIVE FAULT pid=%d signal=%d address=%p\n", getpid(), signal_number, info->si_addr);
#if defined(__x86_64__)
    ucontext_t *state = context;
    dprintf(2, "REGISTERS RIP=%llx RAX=%llx RDI=%llx RSI=%llx RDX=%llx RCX=%llx R8=%llx\n",
            (unsigned long long)state->uc_mcontext.gregs[REG_RIP],
            (unsigned long long)state->uc_mcontext.gregs[REG_RAX],
            (unsigned long long)state->uc_mcontext.gregs[REG_RDI],
            (unsigned long long)state->uc_mcontext.gregs[REG_RSI],
            (unsigned long long)state->uc_mcontext.gregs[REG_RDX],
            (unsigned long long)state->uc_mcontext.gregs[REG_RCX],
            (unsigned long long)state->uc_mcontext.gregs[REG_R8]);
#endif
    void *frames[48];
    int n = backtrace(frames, 48);
    backtrace_symbols_fd(frames, n, 2);
    _exit(128 + signal_number);
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    printf("PROBE PID %d\n", getpid());
    struct sigaction action = {0};
    action.sa_sigaction = crash;
    action.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &action, NULL);
    sigaction(SIGABRT, &action, NULL);
    const char *kind = argc > 1 ? argv[1] : "driver";
    const char *library = argc > 2 ? argv[2] : "libcuda.so.1";
    printf("DLOPEN %s\n", library);
    void *handle = dlopen(library, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) { fprintf(stderr, "%s\n", dlerror()); return 2; }
    const char *symbol = strcmp(kind, "driver") == 0 ? "cuInit" : "cudaGetDeviceCount";
    void *entry = dlsym(handle, symbol);
    if (!entry) { fprintf(stderr, "%s\n", dlerror()); return 3; }
    Dl_info location;
    if (dladdr(entry, &location)) printf("RESOLVED %s -> %s\n", symbol, location.dli_fname);
    if (strcmp(kind, "driver") == 0) {
        int (*version_fn)(int *) = dlsym(handle, "cuDriverGetVersion");
        int version = 0;
        printf("cuDriverGetVersion START\n");
        int result = version_fn(&version);
        printf("cuDriverGetVersion result=%d version=%d\n", result, version);
        int (*init_fn)(unsigned int) = entry;
        printf("cuInit START\n");
        result = init_fn(0);
        printf("cuInit result=%d\n", result);
        if (result) return result;
        int (*count_fn)(int *) = dlsym(handle, "cuDeviceGetCount");
        int count = -1;
        result = count_fn(&count);
        printf("cuDeviceGetCount result=%d count=%d\n", result, count);
        return result;
    }
    int (*count_fn)(int *) = entry;
    int count = -1;
    printf("cudaGetDeviceCount START\n");
    int result = count_fn(&count);
    printf("cudaGetDeviceCount result=%d count=%d\n", result, count);
    return result;
}
