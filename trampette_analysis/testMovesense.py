# -*- coding: utf-8 -*-
"""
Python gatt_sensordata_app client example using the Bleak GATT client.

This example is based on the examples in the Bleak repo: https://github.com/hbldh/bleak
"""

import logging
import asyncio
import platform
import signal
from bleak import BleakClient
from bleak import _logger as logger
from bleak import discover
from functools import reduce
from typing import List
import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ahrs.filters import EKF
from scipy.spatial.transform import Rotation as R


WRITE_CHARACTERISTIC_UUID = (
    "34800001-7185-4d5d-b431-630e7050e8f0"
)

NOTIFY_CHARACTERISTIC_UUID = (
    "34800002-7185-4d5d-b431-630e7050e8f0"
)

FREQUENCY = 104 # IMU frequency in Hz
UPDATE_INTERVAL = 30

# https://stackoverflow.com/a/56243296
class DataView:
    def __init__(self, array, bytes_per_element=1):
        """
        bytes_per_element is the size of each element in bytes.
        By default we are assume the array is one byte per element.
        """
        self.array = array
        self.bytes_per_element = 1

    def __get_binary(self, start_index, byte_count, signed=False):
        integers = [self.array[start_index + x] for x in range(byte_count)]
        _bytes = [integer.to_bytes(
            self.bytes_per_element, byteorder='little', signed=signed) for integer in integers]
        return reduce(lambda a, b: a + b, _bytes)

    def get_uint_16(self, start_index):
        bytes_to_read = 2
        return int.from_bytes(self.__get_binary(start_index, bytes_to_read), byteorder='little')

    def get_uint_8(self, start_index):
        bytes_to_read = 1
        return int.from_bytes(self.__get_binary(start_index, bytes_to_read), byteorder='little')

    def get_uint_32(self, start_index):
        bytes_to_read = 4
        binary = self.__get_binary(start_index, bytes_to_read)
        return struct.unpack('<I', binary)[0]  # <f for little endian

    def get_int_32(self, start_index):
        bytes_to_read = 4
        binary = self.__get_binary(start_index, bytes_to_read)
        return struct.unpack('<i', binary)[0]  # < for little endian

    def get_float_32(self, start_index):
        bytes_to_read = 4
        binary = self.__get_binary(start_index, bytes_to_read)
        return struct.unpack('<f', binary)[0]  # <f for little endian



async def run_queue_consumer(queue: asyncio.Queue):
    while True:
        data = await queue.get()
        if data is None:
            logger.info(
                "Got message from client about disconnection. Exiting consumer loop..."
            )
            break
        #else:
            # print to stdout
            # print(data)

PACKET_TYPE_DATA = 2
PACKET_TYPE_DATA_PART2 = 3
ongoing_data_update = None
update_counter = 0

async def run_ble_client( end_of_serial: str, sensor_types: List[str], queue: asyncio.Queue):

    # Check the device is available
    devices = await discover()
    found = False
    address = None
    for d in devices:
        logger.debug("device:", d)
        if d.name and d.name.endswith(end_of_serial):
            logger.info("device found")
            address = d.address
            found = True
            break

    # This event is set if device disconnects or ctrl+c is pressed
    disconnected_event = asyncio.Event()

    def raise_graceful_exit(*args):
        disconnected_event.set()

    def disconnect_callback(client):
        logger.info("Disconnected callback called!")
        disconnected_event.set()

    async def notification_handler(sender, data):
        """Simple notification handler which prints the data received."""
        d = DataView(data)

        packet_type= d.get_uint_8(0)
        reference = d.get_uint_8(1)
        global q
        global ongoing_data_update
        global update_counter
        # print("packet ", packet_type, ", ongoing:",ongoing_data_update)
        if packet_type == PACKET_TYPE_DATA:
            # ECG (reference 100) fits in one packet
            if reference == 100:
                timestamp = d.get_uint_32(2)
            
                for i in range(0,16):
                    # Interpolate timestamp within the data notification
                    row_timestamp = timestamp + int(i*1000/200)
                    ## ECG package starts with timestamp and then array of 16 samples
                    # Sample scaling is 0.38 uV/sample
                    sample_mV = d.get_int_32(6+i*4) * 0.38 *0.001
                    msg_row = "ECG,{},{:.3f}".format(row_timestamp, sample_mV)
                
                    # queue message for later consumption (output)
                    await queue.put(msg_row)

            else:
            # print("PACKET_TYPE_DATA")
            # Store 1st part of the incoming data
                ongoing_data_update = d
    
        elif packet_type == PACKET_TYPE_DATA_PART2:
            # print("PACKET_TYPE_DATA_PART2. len:",len(data))
                
            # Create combined DataView that contains the whole data packet
            # (skip type_id + ref num of the data_part2)
            d = DataView( ongoing_data_update.array + data[2:])
            ongoing_data_update = None

            # Extract timestamp and IMU values
            timestamp = d.get_uint_32(2)
            for i in range(0, int(FREQUENCY/13)):
                row_timestamp = timestamp + int(i * 1000 / FREQUENCY)
                offset = 6 + i * 3 * 4
                skip = 3 * 8 * 4

                acc = np.array([
                    d.get_float_32(offset),
                    d.get_float_32(offset + 4),
                    d.get_float_32(offset + 8)
                ])
                gyro = np.deg2rad(np.array([
                    d.get_float_32(offset + skip),
                    d.get_float_32(offset + skip + 4),
                    d.get_float_32(offset + skip + 8)
                ]))  # Convert to rad/sec
                mag = np.array([
                    d.get_float_32(offset + 2 * skip),
                    d.get_float_32(offset + 2 * skip + 4),
                    d.get_float_32(offset + 2 * skip + 8)
                ])
                # Update orientation using Madgwick filter
                q = ekf.update(q, gyr=gyro, acc=acc, mag=mag)
                
                # Update the plot every nth data point
                if update_counter % UPDATE_INTERVAL == 0:
                    # Update the 3D plot with the new orientation.
                    update_plot(q)
                    update_counter += 1

                # Output message
                msg_row = "IMU9,{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(
                    row_timestamp, *acc, *gyro, *mag
                )
                await queue.put(msg_row)

            
    if found:
        async with BleakClient(address, disconnected_callback=disconnect_callback) as client:

            # Add signal handler for ctrl+c
            signal.signal(signal.SIGINT, raise_graceful_exit)
            signal.signal(signal.SIGTERM, raise_graceful_exit)
           
            # Start notifications and subscribe to acceleration @ 13Hz
            logger.info("Enabling notifications")
            await client.start_notify(NOTIFY_CHARACTERISTIC_UUID, notification_handler)
            logger.info("Subscribing datastream")
            if "ECG" in sensor_types:
                await client.write_gatt_char(WRITE_CHARACTERISTIC_UUID, bytearray([1, 100])+bytearray("/Meas/ECG/200", "utf-8"), response=True)
            if "IMU9" in sensor_types:
                await client.write_gatt_char(WRITE_CHARACTERISTIC_UUID, bytearray([1, 99])+bytearray(f"/Meas/IMU9/{FREQUENCY}", "utf-8"), response=True)

            # Run until disconnect event is set
            await disconnected_event.wait()
            logger.info(
                "Disconnect set by ctrl+c or real disconnect event. Check Status:")

            # Check the conection status to infer if the device disconnected or crtl+c was pressed
            status = client.is_connected
            logger.info("Connected: {}".format(status))

            # If status is connected, unsubscribe and stop notifications
            if status:
                logger.info("Unsubscribe")
                await client.write_gatt_char(WRITE_CHARACTERISTIC_UUID, bytearray([2, 99]), response=True)
                await client.write_gatt_char(WRITE_CHARACTERISTIC_UUID, bytearray([2, 100]), response=True)
                logger.info("Stop notifications")
                await client.stop_notify(NOTIFY_CHARACTERISTIC_UUID)
            
            # Signal consumer to exit
            await queue.put(None)
            
            await asyncio.sleep(1.0)

    else:
        # Signal consumer to exit
        await queue.put(None)
        print("Sensor  ******" + end_of_serial, "not found!")

# Initialize Madgwick filter
ekf = EKF()
# Starting with an identity quaternion: [scalar, x, y, z]
q = np.array([1.0, 0.0, 0.0, 0.0])

# Set up real-time visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plt.ion()  # interactive mode on
plt.show()

def update_plot(q):
    """
    Updates the 3D plot by drawing the sensor's coordinate frame.
    The sensor's axes (X, Y, Z) are rotated according to the EKF output quaternion.
    
    Parameters:
      q (array): Quaternion [q0, q1, q2, q3] from the EKF, with q0 as the scalar.
    """
    # Convert quaternion from [q0, q1, q2, q3] to [q1, q2, q3, q0] for scipy
    quat_for_scipy = np.array([q[1], q[2], q[3], q[0]])
    R_matrix = R.from_quat(quat_for_scipy).as_matrix()
    
    # Define sensor frame unit vectors in sensor coordinates
    origin = np.array([0, 0, 0])
    x_axis = np.array([1, 0, 0])  # Sensor X-axis
    y_axis = np.array([0, 1, 0])  # Sensor Y-axis
    z_axis = np.array([0, 0, 1])  # Sensor Z-axis

    # Rotate sensor axes to global frame using the rotation matrix
    x_rot = R_matrix @ x_axis
    y_rot = R_matrix @ y_axis
    z_rot = R_matrix @ z_axis

    # Clear the previous frame and draw new vectors
    ax.clear()
    ax.quiver(*origin, *x_rot, color="r", label="Sensor X")
    ax.quiver(*origin, *y_rot, color="g", label="Sensor Y")
    ax.quiver(*origin, *z_rot, color="b", label="Sensor Z")

    # Set global frame axis limits and labels
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel("Global X")
    ax.set_ylabel("Global Y")
    ax.set_zlabel("Global Z")
    ax.set_title("Sensor Frame in Global Coordinates")
    plt.legend()
    plt.draw()
    plt.pause(0.001)


async def main(end_of_serial: str, sensor_types: List[str]):

    queue = asyncio.Queue()

    client_task = run_ble_client(end_of_serial, sensor_types, queue)
    consumer_task = run_queue_consumer(queue)
    await asyncio.gather(client_task, consumer_task)
    logger.info("Main method done.")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # print usage if command line arg not given
    if len(sys.argv)<2:
        print("Usage: python movesense_sensor_data <end_of_sensor_name> <sensor_type>")
        print("sensor_type must be either 'IMU9', 'ECG', or omitted to run both")
        exit(1)
    end_of_serial = sys.argv[1]
    sensor_type = sys.argv[2] if len(sys.argv) > 2 else ""
    

    # Ensure valid sensor type and run the corresponding function
    if sensor_type == "":
        sensor_types = ["IMU9", "ECG"]
    elif sensor_type in ["IMU9", "ECG"]:
        sensor_types = [sensor_type]
    else:
        print("Error: sensor_type must be either 'IMU9' or 'ECG'")
        exit(1)
    asyncio.run(main(end_of_serial, sensor_types))
