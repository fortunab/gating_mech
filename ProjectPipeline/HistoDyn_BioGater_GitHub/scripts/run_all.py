import subprocess
import sys


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    models = ["biogater", "convmixer", "depthwise"]
    for model in models:
        run([sys.executable, "scripts/train.py", "--model", model])


if __name__ == "__main__":
    main()
