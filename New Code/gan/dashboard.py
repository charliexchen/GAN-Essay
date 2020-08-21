import pygame
import numpy as np
import math
import random


class histogram:
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

    def draw_histogram(self, screen, colour):
        display = pygame.display.get_surface()
        width = display.get_width()
        height = display.get_height()
        for ind, bucket in enumerate(self.scaled_histogram()):
            rectangle_height = bucket * height
            rectangle_width = width / self.bucket_count
            rect = pygame.Rect(ind * rectangle_width
                               , height, rectangle_width, -rectangle_height)
            pygame.draw.rect(screen, colour, rect)


class graph:
    "function to creat realtime training graph"


if __name__ == "__main__":

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    boundary = [200, 200]
    pygame.init()
    screen = pygame.display.set_mode(boundary)
    clock = pygame.time.Clock()
    done = False
    hist = histogram(20, -2, 2)
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # If user clicked close
                done = True
        for i in range( 100):
            hist.add_datapoint(random.normalvariate(0,1))
        screen.fill(BLACK)
        hist.draw_histogram(screen, WHITE)

        pygame.display.flip()
        clock.tick(30)
