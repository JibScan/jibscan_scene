from pathlib import Path

from jibscan_scene.fixtures import generate_synthetic_squid_rgba


if __name__ == "__main__":
    path = generate_synthetic_squid_rgba(Path("examples/fixtures/synthetic_squid_001.png"))
    print(path)
