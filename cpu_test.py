from pprint import pprint

from PyLibreHardwareMonitor import Computer


def main() -> None:
    computer = Computer()
    cpu_data = computer.cpu

    print("Results:")
    pprint(cpu_data)

    if not cpu_data:
        raise RuntimeError(
            "No CPU sensor data available. "
        )


if __name__ == "__main__":
    main()