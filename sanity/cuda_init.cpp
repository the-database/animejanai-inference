// Two independently linked DLLs must be able to initialize CUDA together.
// Run in a fresh process each time to exercise the hybrid loader's first use.
#include <windows.h>
#include <atomic>
#include <cstdio>
#include <thread>

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::fprintf(stderr, "usage: aji_cuda_init_test <absolute DLL A> <absolute DLL B>\n");
        return 2;
    }
    HMODULE modules[2]{};
    int (*probes[2])(){};
    for (int i = 0; i < 2; ++i) {
        modules[i] = LoadLibraryExA(argv[i+1], nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (modules[i]) probes[i] = (int(*)())GetProcAddress(modules[i], "probe");
        if (!probes[i]) {
            std::fprintf(stderr, "Cannot load probe %d: Windows error %lu\n", i, GetLastError());
            for (HMODULE module : modules) if (module) FreeLibrary(module);
            return 2;
        }
    }
    std::atomic<int> ready{0};
    int first = 1, second = 1;
    std::thread worker([&] {
        ++ready;
        while (ready < 2) std::this_thread::yield();
        first = probes[0]();
    });
    ++ready;
    while (ready < 2) std::this_thread::yield();
    second = probes[1]();
    worker.join();
    for (HMODULE module : modules) FreeLibrary(module);
    if (first || second) return 1;
    std::puts("PASS concurrent CUDA initialization in separate DLLs");
    return 0;
}
