import numpy
from PyQt5.QtCore import QThread, pyqtSignal
import time
import numpy as np
import matplotlib.pyplot as plt

from utils.error_calculation import calculate_frobenius_error

inpainting_times = {}
cumulative_times = {}


class InpaintingThread(QThread):
    finished_signal = pyqtSignal(numpy.ndarray)
    starting_signal = pyqtSignal()

    def __init__(self, inpaint_algorithm, image_getter, mutex, condition, video_inpainting=True, first_time=False):
        QThread.__init__(self)
        self.video_inpainting = video_inpainting
        self.inpaint_algorithm = inpaint_algorithm
        self.image_getter = image_getter
        self.image_mutex = mutex
        self.image_condition = condition
        self.inpainting_times = inpainting_times
        self.cumulative_times = cumulative_times
        self.first_time = first_time

    def run(self):
        iteration_count = 0  # Add a counter for iterations
        while not self.isInterruptionRequested():
            start_time = time.time()
            self.starting_signal.emit()

            image_path, mask_path = self.image_getter()
            output = self.inpaint_algorithm.inpaint(image_path, mask_path)

            error = calculate_frobenius_error(image_path, output, mask_path, self.inpaint_algorithm.is_deep_learning)

            print(f"Frobenius error: {error}")

            self.image_mutex.lock()
            self.finished_signal.emit(output)
            self.image_condition.wait(self.image_mutex)
            self.image_mutex.unlock()

            end_time = time.time()
            current_algorithm = self.inpaint_algorithm.__class__.__name__

            # Only append times if it's not the first or last iteration
            if iteration_count != 0 and not self.isInterruptionRequested() or not self.video_inpainting and not self.first_time:
                if current_algorithm not in self.inpainting_times:
                    self.inpainting_times[current_algorithm] = []
                    self.cumulative_times[current_algorithm] = 0
                self.cumulative_times[current_algorithm] += end_time - start_time
                self.inpainting_times[current_algorithm].append((end_time - start_time))

            iteration_count += 1
            print("Whole Inpainting took: " + str(end_time - start_time))
            if self.video_inpainting is False:
                return

    def stop(self):
        self.requestInterruption()  # Request the thread to stop
        self.image_condition.wakeAll()  # Wake up the thread if it's waiting

    def show_graph(self):
        plt.figure()
        has_data = False  # Flag to check if any data was plotted

        for algo, times in self.inpainting_times.items():
            if times:  # Check if there are times for the current algorithm
                cumulative_times = np.cumsum(times)  # Calculate the cumulative sum of the times for plotting
                plt.plot(cumulative_times, times, marker='o', label=algo)
                has_data = True

        if not has_data:
            plt.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes)
        else:
            plt.xlabel('Cumulative Running Time (s)')
            plt.ylabel('Time taken for Each Inpainting (s)')
            plt.title('Inpainting Performance Comparison')
            plt.legend()
            plt.grid(True)

        plt.show()
