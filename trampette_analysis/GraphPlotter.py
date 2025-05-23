import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import numpy as np

class GraphPlotter:
    """
    A class for plotting 2D and 3D graphs using an arbitrary number of data series.
    
    - The plot2d method will plot all provided series against the sample index.
    - The plot3d method will plot a 3D trajectory from three chosen series.
    """
    def __init__(self, **series):
        """
        Initialize the GraphPlotter with an arbitrary number of data series.

        Pass the series as keyword arguments where the key is the label
        and the value is the data vector (list, NumPy array, etc.).

        Example:
            plotter = GraphPlotter(Temperature=temp_data, Pressure=pressure_data)
        """
        self.series = {}
        for label, data in series.items():
            # Convert each series to a NumPy array.
            self.series[label] = np.array(data)
        
        # Ensure that all series have the same length.
        lengths = [len(data) for data in self.series.values()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All series must have the same length.")

    def plot2d(self, title="2D Data Series Plot", xlabel="Sample Index", ylabel="Value", figsize=(10, 6)):
        """
        Plot all the data series in a single 2D plot using the sample index as the x-axis.
        
        Parameters:
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            figsize (tuple): Size of the figure (width, height).
        """
        plt.figure(figsize=figsize)
        x_axis = np.arange(len(next(iter(self.series.values()))))  # Using length from any series.
        for label, data in self.series.items():
            plt.plot(x_axis, data, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot3d(self, x_key=None, y_key=None, z_key=None, title="3D Trajectory", xlabel="X", ylabel="Y", zlabel="Z", figsize=(10, 8)):
        """
        Plot a 3D trajectory using three chosen data series.

        You can specify which series to use by providing the keys (as passed during initialization)
        for the x, y, and z axes. If no keys are provided and exactly three series are present,
        they will be used in the order of insertion.

        Parameters:
            x_key (str): Key for the series to use as the x-axis.
            y_key (str): Key for the series to use as the y-axis.
            z_key (str): Key for the series to use as the z-axis.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            zlabel (str): Label for the z-axis.
            figsize (tuple): Size of the figure (width, height).

        Raises:
            ValueError: If three series cannot be determined for a 3D plot.
        """
        # Determine which series to use for the 3D plot.
        if x_key and y_key and z_key:
            try:
                x_data = self.series[x_key]
                y_data = self.series[y_key]
                z_data = self.series[z_key]
            except KeyError as e:
                raise ValueError(f"Key {e} not found in the provided data series.")
        else:
            # If no keys are specified, use the first three series if available.
            if len(self.series) != 3:
                raise ValueError("Please provide x_key, y_key, and z_key for the 3D plot "
                                 "if you did not initialize exactly three data series.")
            keys = list(self.series.keys())
            x_data = self.series[keys[0]]
            y_data = self.series[keys[1]]
            z_data = self.series[keys[2]]
        
        # Create the 3D plot.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_data, y_data, z_data, label="3D Trajectory", marker='o')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    # Generate some example data
    t = np.linspace(0, 10, 100)
    sin_data = np.sin(t)
    cos_data = np.cos(t)
    linear_data = t
    quadratic_data = t**2

    # Example 1: Using multiple series for 2D plot.
    plotter1 = GraphPlotter(Sine=sin_data, Cosine=cos_data, Linear=linear_data, Quadratic=quadratic_data)
    plotter1.plot2d(title="Multiple Data Series", xlabel="Index", ylabel="Value")

    # Example 2: Using exactly three series for 3D plot (automatic selection).
    plotter2 = GraphPlotter(X=sin_data, Y=cos_data, Z=linear_data)
    plotter2.plot3d(title="3D Trajectory (Automatic Selection)")

    # Example 3: Using more than three series, but specifying which to plot in 3D.
    plotter3 = GraphPlotter(Series1=sin_data, Series2=cos_data, Series3=linear_data, Extra=quadratic_data)
    # Specify the keys to use for the 3D plot:
    plotter3.plot3d(x_key="Series1", y_key="Series2", z_key="Series3", 
                    title="3D Trajectory (Specified Series)", xlabel="Sine", ylabel="Cosine", zlabel="Linear")
