import sys
import zipfile


def main() -> int:
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "-Z1":
        zip_path = " ".join(args[1:])
        with zipfile.ZipFile(zip_path, "r") as archive:
            for name in archive.namelist():
                sys.stdout.write(name + "\n")
        return 0

    if len(args) >= 3 and args[0] == "-p":
        zip_path = " ".join(args[1:-1])
        entry_name = args[-1]
        with zipfile.ZipFile(zip_path, "r") as archive:
            sys.stdout.buffer.write(archive.read(entry_name))
        return 0

    sys.stderr.write("Supported usage: unzip -Z1 <zip> or unzip -p <zip> <entry>\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
