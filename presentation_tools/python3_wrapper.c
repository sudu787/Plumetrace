#include <process.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    char **child_argv;
    int i;
    intptr_t result;
    const char *python_path = "C:\\Users\\sudu\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";

    child_argv = (char **)calloc((size_t)argc + 1, sizeof(char *));
    if (child_argv == NULL) {
        fprintf(stderr, "Unable to allocate python3 wrapper arguments.\n");
        return 1;
    }

    child_argv[0] = (char *)python_path;
    for (i = 1; i < argc; i++) {
        child_argv[i] = argv[i];
    }
    child_argv[argc] = NULL;

    result = _spawnv(_P_WAIT, python_path, (const char * const *)child_argv);
    free(child_argv);

    if (result == -1) {
        fprintf(stderr, "Unable to start bundled Python runtime.\n");
        return 1;
    }

    return (int)result;
}
