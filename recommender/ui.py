"""Console helpers shared by all modules."""
import textwrap

WIDTH = 88


def banner(title):
    print("\n" + "=" * WIDTH)
    print(title.center(WIDTH))
    print("=" * WIDTH)


def section(title):
    print("\n" + "-" * WIDTH)
    print(f" {title}")
    print("-" * WIDTH)


def kv(key, value, width=20):
    lines = textwrap.wrap(str(value), WIDTH - width - 4) or [""]
    print(f"  {key:<{width}}: {lines[0]}")
    for line in lines[1:]:
        print(" " * (width + 4) + line)


def wrap(text, indent=2):
    pad = " " * indent
    print(textwrap.fill(str(text), WIDTH, initial_indent=pad, subsequent_indent=pad))


def info(msg):
    print(f"  [i] {msg}")


def success(msg):
    print(f"  [OK] {msg}")


def warn(msg):
    print(f"  [!] {msg}")
