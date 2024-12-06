import math
# Constants
g = 9.81

# Absolute uncertainties
sigma_dh = 0.01
sigma_ToF = 1.0/104 # this is the frequency of the IMU data
sigma_g = 0.02 # recommended if global value is used

def jumpheight(ToF, dh):
    '''
    Function that calculates the jump height of a gymnast given the time of flight (ToF)
    '''
    totalHeight = (g / (8 * ToF**2)) * (ToF**2 + 2 * dh / g)**2
    # Partial derivatives
    partial_dh = 0.5 + dh / (g * ToF**2)
    partial_ToF = (g * ToF / 4) - (dh**2 / (g * ToF**3))
    partial_g = (ToF**2 / 8) - (dh**2 / (2 * g**2 * ToF**2))
    # Propagated uncertainty
    sigma_h_max = math.sqrt(
        (partial_dh * sigma_dh)**2 +
        (partial_ToF * sigma_ToF)**2 +
        (partial_g * sigma_g)**2
    )
    return (totalHeight, sigma_h_max)

def main():
    # Measured values
    dhs = [0.2] # Test different height differences
    ToFs = [1.5] # Time of flight

    for ToF in ToFs:
        for dh in dhs:
            # Total height and error
            (totalHeight, sigma_h_max) = jumpheight(ToF, dh)
            print(f"With ToF={ToF} and dh={dh} the total height is {totalHeight}m")
            print(f"Absolute Uncertainty: {sigma_h_max}m")
            print(f"Relative Uncertainty: {100*sigma_h_max/totalHeight}%")

if __name__ == "__main__":
    main()
