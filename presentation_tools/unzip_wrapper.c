#include <process.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>

int main(int argc, char **argv) {
    char exe_path[MAX_PATH];
    char script_path[MAX_PATH];
    char *last_slash;
    char **child_argv;
    int i;
    intptr_t result;

    if (GetModuleFileNameA(NULL, exe_path, MAX_PATH) == 0) {
        fprintf(stderr, "Unable to locate unzip wrapper executable.\n");
        return 1;
    }

    strncpy(script_path, exe_path, MAX_PATH - 1);
    script_path[MAX_PATH - 1] = '\0';
    last_slash = strrchr(script_path, '\\');
    if (last_slash == NULL) {
        fprintf(stderr, "Unable to resolve unzip.py path.\n");
        return 1;
    }
    *(last_slash + 1) = '\0';
    strncat(script_path, "unzip.py", MAX_PATH - strlen(script_path) - 1);

    child_argv = (char **)calloc((size_t)argc + 2, sizeof(char *));
    if (child_argv == NULL) {
        fprintf(stderr, "Unable to allocate unzip wrapper arguments.\n");
        return 1;
    }

    child_argv[0] = "C:\\Users\\sudu\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
    child_argv[1] = script_path;
    for (i = 1; i < argc; i++) {
        child_argv[i + 1] = argv[i];
    }
    child_argv[argc + 1] = NULL;

    result = _spawnv(_P_WAIT, child_argv[0], (const char * const *)child_argv);
    free(child_argv);

    if (result == -1) {
        fprintf(stderr, "Unable to start Python for unzip wrapper.\n");
        return 1;
    }

    return (int)result;
}
