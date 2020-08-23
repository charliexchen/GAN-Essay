import pygame
import numpy as np
import math
from colours import *
from gan import GAN
import random
from collections import deque


def get_screen_dims():
    display = pygame.display.get_surface()
    width = display.get_width()
    height = display.get_height()
    return width, height


class Histogram:
    """Function to create realtime histogram and functions"""

    def __init__(self, bucket_count: int, range_min: float, range_max: float):
        assert range_max > range_min, "Range max lower than min"
        self.bucket_count = bucket_count
        self.bucket_size = (range_max - range_min) / self.bucket_count
        self.range_min = range_min
        self.buckets = [0 for _ in range(bucket_count)]
        self.count = 0

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
        width, height = get_screen_dims()
        for ind, bucket in enumerate(self.scaled_histogram()):
            rectangle_height = bucket * height
            rectangle_width = width / self.bucket_count
            rect = pygame.Rect(ind * rectangle_width
                               , height, rectangle_width, -rectangle_height)
            pygame.draw.rect(target_screen, colour, rect)


class Graph:
    """function to creat realtime training graph"""

    def __init__(self, bucket_count: int, range_min: float, range_max: float):
        self.func_inputs = np.linspace(range_min, range_max, bucket_count)
        self.bucket_count = bucket_count

    def draw_function(self, target_screen, colour, func, scale=1.0):
        width, height = get_screen_dims()
        func_outputs = [height - int(func(x) * scale * height) for x in self.func_inputs]

        drawing_points = list(zip(
            np.linspace(0, width, self.bucket_count),
            func_outputs
        ))
        pygame.draw.lines(target_screen, colour, False, drawing_points)

    def draw_vectorised_function(self, target_screen, colour, func, scale=1.0):
        width, height = get_screen_dims()
        func_outputs = func(self.func_inputs)

        scaled_outputs = [height - int(x * scale * height) for x in func_outputs]

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
                # Create rule of thumb bandwidth if non is assigned
                sigma = np.var(list(self.datapoints))
                bandwidth = sigma * ((4 / (3 * n)) ** 0.2)
        return sum(GaussianKernel.gaussian(x, y, bandwidth) for y in list(self.datapoints)) / n

    @staticmethod
    def gaussian(x, mu=0.0, sigma=1.0):
        return math.exp(-(((x - mu) / sigma) ** 2) / 2) / (math.sqrt(2 * math.pi) * sigma)


if __name__ == "__main__":

    gan = GAN()


    def eval_disc(x):
        y = gan.discriminator.predict(x)
        return y.flatten()


    est = GaussianKernel(1024)


    def optimal_disc(x):
        p_fake = est.evaluate(x)
        p_real = GaussianKernel.gaussian(x)
        return p_real / (p_fake + p_real)


    boundary = [500, 200]
    pygame.init()
    screen = pygame.display.set_mode(boundary)
    clock = pygame.time.Clock()
    done = False
    hist = Histogram(100, -6, 6)
    gra = Graph(50, -6, 6)

    count = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # If user clicked close
                done = True
        screen.fill(BLACK)
        samples = gan.train(++count).flatten()

        hist.reset()
        est.add_points(samples)
        hist.add_datapoints(samples)
        hist.draw_histogram(screen, WHITE)
        gra.draw_function(screen, RED, GaussianKernel.gaussian)

        gra.draw_function(screen, GREEN, optimal_disc)
        gra.draw_vectorised_function(screen, BLUE, eval_disc)
        pygame.display.flip()
        clock.tick(30)
