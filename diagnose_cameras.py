# -*- coding: utf-8 -*-
"""Print MVS SDK status and all enumerated cameras."""

from hikrobot_camera import HikRobotService


def main() -> None:
    service = HikRobotService()
    try:
        print("SDK ready:", service.sdk_ready)
        print("SDK version:", hex(service.sdk_version) if service.sdk_version else "")
        print("MvImport:", service.mvimport_path)
        print("Runtime:", service.runtime_path)
        count = service.refresh_devices(quiet=True)
        print("Device count:", count)
        for serial, device in service.devices.items():
            print(f"{device['model']}  SN={serial}  IP={device['ip']}")
        if service.sdk_error:
            print("SDK error:", service.sdk_error)
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
