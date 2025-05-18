#!/usr/bin/env python3
"""
Clean up stray POSIX shared-memory segments created with multiprocessing.shared_memory
or posix_ipc.  Works on any Linux where segments appear as regular files in /dev/shm.

Author: you
"""

import os, sys, argparse, textwrap
from pathlib import Path
from multiprocessing import shared_memory

SHM_DIR = Path("/dev/shm")

def owned_segments(prefix: str | None = None):
    """Yield (name, size_bytes) for shm objects owned by the current user."""
    uid = os.getuid()
    for entry in SHM_DIR.iterdir():
        if entry.stat().st_uid != uid:
            continue                      # not yours
        if prefix and not entry.name.startswith(prefix):
            continue
        yield entry.name, entry.stat().st_size

def unlink(name: str):
    """Unlink - and close if we can attach – the shared-memory object *name*."""
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.unlink()          # marks for removal
        shm.close()           # close our fd
        print(f"  ✔ unlinked {name!r}")
    except FileNotFoundError:
        print(f"  ⚠ {name!r} already gone")
    except Exception as e:
        print(f"  ✗ could not unlink {name!r}: {e}")

def main():
    desc = """\
    List and optionally remove POSIX shared-memory segments owned by you.
    A segment's *name* is what you passed to SharedMemory(name=…).
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(desc)
    )
    parser.add_argument("-a", "--all",   action="store_true",
                        help="select *every* segment you own (respecting --prefix)")
    parser.add_argument("-p", "--prefix", default=None,
                        help="only touch segments whose names start with PREFIX")
    parser.add_argument("-f", "--force", action="store_true",
                        help="do not ask for confirmation")
    args = parser.parse_args()

    segs = list(owned_segments(args.prefix))
    if not segs:
        print("No matching shared-memory objects found.")
        return

    print("Found the following segments:")
    for n, sz in segs:
        print(f"  {n:<32} {sz/1024:.1f} KiB")

    # build deletion list
    to_delete = segs if args.all else []

    if not args.all:
        # interactive selection
        while True:
            choice = input("\nEnter name to delete (or blank to finish): ").strip()
            if not choice:
                break
            matches = [s for s in segs if s[0] == choice]
            if not matches:
                print("  name not in list.")
            else:
                to_delete.append(matches[0])

    if not to_delete:
        print("Nothing to remove.")
        return

    if not args.force:
        print("\nSegments selected for removal:")
        for n, _ in to_delete:
            print("  ", n)
        resp = input("Proceed? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    print("\nRemoving …")
    for name, _ in to_delete:
        unlink(name)

if __name__ == "__main__":
    main()
