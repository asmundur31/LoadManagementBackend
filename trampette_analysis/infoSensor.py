import asyncio
import json
from bleak import BleakScanner, BleakClient, BleakError

# Movesense UUIDs from the official example
MOVESENSE_SERVICE_UUID = "0000fdf3-0000-1000-8000-00805f9b34fb"
WRITE_UUID = "6b200001-ff4e-4979-8186-fb7ba486fcd7"  # Command UUID
NOTIFY_UUID = "6b200002-ff4e-4979-8186-fb7ba486fcd7"  # Data UUID

async def scan_sensors():
    """Scans for available Movesense sensors and returns a list of them."""
    print("🔍 Scanning for Movesense sensors...")
    devices = await BleakScanner.discover()
    movesense_devices = [device for device in devices if device.name and "Movesense" in device.name]

    if not movesense_devices:
        print("❌ No Movesense sensors found. Make sure the device is turned on.")
        return None

    print("\n📡 Available Movesense Sensors:")
    for idx, device in enumerate(movesense_devices):
        print(f"[{idx}] {device.name} ({device.address})")

    while True:
        try:
            choice = int(input("\n👉 Select a sensor by number: "))
            if 0 <= choice < len(movesense_devices):
                return movesense_devices[choice]
            else:
                print("❌ Invalid selection. Try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

async def imu_notification_handler(sender, data):
    """Handles incoming IMU sensor data notifications."""
    try:
        print(f"✅ IMU Data Received: {data.decode('utf-8')}")
    except Exception as e:
        print(f"⚠️ Error decoding IMU data: {e}")

async def get_sensor_info(device_address):
    """Connects to the sensor, subscribes to IMU data, and fetches metadata."""
    async with BleakClient(device_address) as client:
        print(f"\n✅ Connected to Movesense: {device_address}\n")

        # 🔹 Fetch and print available services and characteristics
        print("🔍 Fetching services and characteristics...\n")
        for service in client.services:
            print(f"🔹 Service: {service.uuid}")
            for char in service.characteristics:
                print(f"   ↳ Characteristic: {char.uuid} (Properties: {char.properties})")

        # 🔹 Start IMU Data Streaming
        imu_start_command = json.dumps({"Uri": "/Meas/IMU9/52", "Method": "PUT"}).encode("utf-8")
        try:
            await client.write_gatt_char(WRITE_UUID, imu_start_command, response=False)
            print("📡 Sent command to start IMU streaming...")
        except Exception as e:
            print(f"⚠️ Could not start IMU streaming: {e}")

        # 🔹 Subscribe to IMU Data Notifications
        try:
            await client.start_notify(NOTIFY_UUID, imu_notification_handler)
            print("📡 Subscribed to IMU data notifications... Receiving data.")
            await asyncio.sleep(10)  # Keep receiving for 10 seconds
            await client.stop_notify(NOTIFY_UUID)
        except Exception as e:
            print(f"⚠️ Could not subscribe to IMU data: {e}")

async def main():
    selected_device = await scan_sensors()
    if selected_device:
        await get_sensor_info(selected_device.address)

asyncio.run(main())
