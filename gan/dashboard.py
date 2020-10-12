import math
import numpy as np
import pygame
import random
import sys
import yaml
from collections import deque
from colours import *
from gan_builder import GANBuilder


def get_screen_dims():
    display = pygame.display.get_surface()
    width = display.get_width()
    height = display.get_height()
    return width, height


class Histogram:
    """Function to create realtime histogram and functions"""

    def __init__(self, bucket_count: int, range_min: float, range_max: float, offset: float = 0.0):
        assert range_max > range_min, "Range max lower than min"
        self.bucket_count = bucket_count
        self.bucket_size = (range_max - range_min) / self.bucket_count
        self.range_min = range_min
        self.buckets = [0 for _ in range(bucket_count)]
        self.count = 0
        self.offset = offset

    def reset(self):
        self.buckets = [0 for _ in range(self.bucket_count)]
        self.count = 0

    def add_datapoint(self, new_point):
        bucket_index = math.floor((new_point - self.range_min) / self.bucket_size)
        self.count += 1
        if bucket_index in range(self.bucket_count):
            self.buckets[bucket_index] += 1

    def add_datapoints(self, new_points):
        for point in new_points:
            self.add_datapoint(point)

    def scaled_histogram(self, scale: float = 1):
        return [scale * x / (self.bucket_size * self.count) for x in self.buckets]

    def draw_histogram(self, target_screen, colour):
        # offset determines where on the screen is the zero vertically i.e. 0 means histogram sits on bottom of screen
        width, height = get_screen_dims()
        histogram_max_height = int((1 - self.offset) * height)
        for ind, bucket in enumerate(self.scaled_histogram()):
            rectangle_height = bucket * histogram_max_height
            rectangle_width = width / self.bucket_count
            rect = pygame.Rect(ind * rectangle_width
                               , histogram_max_height, rectangle_width, -rectangle_height)
            pygame.draw.rect(target_screen, colour, rect)


class Graph:
    """function to create realtime training graph"""

    def __init__(self, bucket_count: int, range_min: float, range_max: float, offset: float = 0):
        self.func_inputs = np.linspace(range_min, range_max, bucket_count)
        self.bucket_count = bucket_count
        self.offset = offset

    def draw_function(self, target_screen, colour, func, scale=1.0):
        width, height = get_screen_dims()
        max_height = int(height * (1 - self.offset))
        func_outputs = [max_height - int(func(x) * scale * max_height) for x in self.func_inputs]

        drawing_points = list(zip(
            np.linspace(0, width, self.bucket_count),
            func_outputs
        ))
        pygame.draw.lines(target_screen, colour, False, drawing_points)

    def draw_vectorised_function(self, target_screen, colour, func, scale=1.0):
        width, height = get_screen_dims()
        max_height = int(height * (1 - self.offset))
        func_outputs = func(self.func_inputs)
        scaled_outputs = [max_height - int(x * scale * max_height) for x in func_outputs]
        drawing_points = list(zip(
            np.linspace(0, width, self.bucket_count),
            scaled_outputs
        ))
        pygame.draw.lines(target_screen, colour, False, drawing_points)


class GaussianKernel:
    """Function for smoothing input function"""

    def __init__(self, max_len=None):
        self.datapoints = deque([], max_len)

    def add_point(self, point):
        self.datapoints.append(point)

    def add_points(self, points):
        self.datapoints.extend(points)

    def evaluate(self, x, bandwidth=None):

        n = len(self.datapoints)
        if bandwidth is None:
            if n < 5:
                bandwidth = 1.0
            else:
                # Create rule of thumb  bandwidth if non is assigned
                sigma = np.var(list(self.datapoints))
                bandwidth = sigma * ((4 / (3 * n)) ** 0.2)
        return sum(GaussianKernel.gaussian(x, y, bandwidth) for y in list(self.datapoints)) / n

    @staticmethod
    def gaussian(x, mu=0.0, sigma=1.0):
        return math.exp(-(((x - mu) / sigma) ** 2) / 2) / (math.sqrt(2 * math.pi) * sigma)


class GanWithDashboard:
    def __init__(self, config_path=None):
        if config_path is None:
            raise IOError("No config file specified")
        config = self._get_config(config_path)
        self.gan = GANBuilder(config=config)
        self.dashboard_config = config['dashboard_config']
        self.histogram = None
        self.graph = None
        plot_range = self.dashboard_config['plot_range']
        histogram_config = self.dashboard_config['histogram_config']
        if histogram_config['show_histogram']:
            self.histogram = Histogram(histogram_config['buckets'], plot_range[0], plot_range[1],
                                       self.dashboard_config['offset'])

        graph_config = self.dashboard_config['graph_config']
        if graph_config['show_graphs'] and graph_config['points'] > 0:
            self.graph = Graph(graph_config['points'], plot_range[0], plot_range[1], self.dashboard_config['offset'])

        if graph_config['show_generator_kernel']:
            self.gaussian_estimator = GaussianKernel(1024)

    @staticmethod
    def _get_config(config_path):
        with open(config_path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def show_graph(self):

        def optimal_disc(x):
            p_fake = self.gaussian_estimator.evaluate(x)
            p_real = GaussianKernel.gaussian(x)
            return p_real / (p_fake + p_real)

        def eval_disc(x):
            y = gan.discriminator.predict(x)
            return y.flatten()

        gan = GANBuilder(config_filename)
        pygame.init()
        screen = pygame.display.set_mode(self.dashboard_config['screen_size'])
        graph_config = self.dashboard_config['graph_config']
        clock = pygame.time.Clock()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # If user clicked close
                    done = True
            screen.fill(BLACK)
            samples = gan.train_one_batch()

            self.histogram.reset()
            self.histogram.add_datapoints(samples)
            self.histogram.draw_histogram(screen, WHITE)

            if graph_config['show_generator_kernel']:
                self.gaussian_estimator.add_points(samples)
                if graph_config['show_optimal_discriminator_kernel']:
                    self.graph.draw_function(screen, GREEN, optimal_disc)
            if graph_config['show_optimal_generator']:
                self.graph.draw_function(screen, RED, GaussianKernel.gaussian)
            if graph_config['show_discriminator']:
                self.graph.draw_vectorised_function(screen, BLUE, eval_disc)

            pygame.display.flip()
            clock.tick(30)


if __name__ == "__main__":
    config_filename = sys.argv[1]
    gan_with_dashboard = GanWithDashboard(config_filename)

    gan_with_dashboard.show_graph()
