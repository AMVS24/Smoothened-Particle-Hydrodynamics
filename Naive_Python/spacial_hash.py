import random
import os
import numpy as np
import pyglet
import math
import time as CPUTime
from pyglet import clock
import sys

screen_width = 800
screen_height = 600

ORIGIN = np.array([screen_width / 2, screen_height / 2])
win = pyglet.window.Window(screen_width, screen_height, caption="Balls")

main_batch = pyglet.graphics.Batch()
dt = 0.01

# box_attributes = [600, 400, 5]
box_attributes = [600, 400, 5]  # width height thickness

box_b = pyglet.shapes.Rectangle(0.5 * (screen_width - 2 * box_attributes[2] - box_attributes[0]),
                                0.5 * (screen_height - 2 * box_attributes[2] - box_attributes[1]),
                                box_attributes[0] + 2 * box_attributes[2], box_attributes[2], batch=main_batch)
box_l = pyglet.shapes.Rectangle(0.5 * (screen_width - 2 * box_attributes[2] - box_attributes[0]),
                                0.5 * (screen_height - 2 * box_attributes[2] - box_attributes[1]),
                                box_attributes[2], box_attributes[1] + 2 * box_attributes[2], batch=main_batch)
box_t = pyglet.shapes.Rectangle(0.5 * (screen_width - 2 * box_attributes[2] - box_attributes[0]),
                                0.5 * (screen_height - 0 * box_attributes[2] + box_attributes[1]),
                                box_attributes[0] + 2 * box_attributes[2], box_attributes[2], batch=main_batch)
box_r = pyglet.shapes.Rectangle(0.5 * (screen_width - 0 * box_attributes[2] + box_attributes[0]),
                                0.5 * (screen_height - 2 * box_attributes[2] - box_attributes[1]),
                                box_attributes[2], box_attributes[1] + 2 * box_attributes[2], batch=main_batch)


class SPHSim:
    def __init__(self, h, pressure_multiplier, target_density, ball_count):
        self.smoothening_radius = h
        self.pressure_multiplier = pressure_multiplier
        self.target_density = target_density
        self.ball_count = ball_count
        self.cached_distances = np.zeros((ball_count, ball_count))
        self.cell_count = int(
            (box_attributes[0] / self.smoothening_radius) * (box_attributes[1] / self.smoothening_radius))
        self.start_indices = {}
        self.acceleration_array = np.zeros((self.ball_count, 2), dtype=float)
        self.velocity_array = np.zeros((self.ball_count, 2), dtype=float)
        self.position_array = np.zeros((self.ball_count, 2), dtype=float)
        self.density_array = np.zeros(self.ball_count, dtype=float)
        self.distance_count = 0

    def update_parameters(self):
        self.position_array += self.velocity_array * dt
        self.velocity_array += self.acceleration_array * dt  # Currently utilising Euler Integration


if (len(sys.argv) > 1): 
    c = int(sys.argv[1])
else:
    c=1000
Sim = SPHSim(20, 2000, 1, c)


class ball():
    def __init__(self, radius, number, m=20, batch=main_batch, name=None):
        self.m = m
        self.name = name
        self.no = number
        self.r = radius
        self.sprite = pyglet.shapes.Circle(*(ORIGIN + self.p()), radius, color=(0, 0, 255), batch=batch)

    def update_parameters(self):
        self.sprite.position = ORIGIN + self.p()

    def p(self):
        return Sim.position_array[self.no]

    def v(self):
        return Sim.velocity_array[self.no]

    def a(self):
        return Sim.acceleration_array[self.no]

    def density(self):
        return Sim.density_array[self.no]


ball_list = []
for i in range(Sim.ball_count):
    rand_x = random.uniform(-290, 290)
    rand_y = random.uniform(-190, 190)
    rand_v = (100 * random.uniform(-1, 1), 10 * random.uniform(-1, 1))
    Sim.position_array[i] = np.array((rand_x, rand_y))
    Sim.velocity_array[i] = np.array(rand_v)
    ball_list.append(ball(20, i))

ball_count = len(ball_list)


def vel_flip(normal, v1, v2, drag=0.85):
    return v1 - (normal * (np.dot(normal, v1) - np.dot(normal, v2))) * drag


def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros_like(vector)
    return vector / norm


def check_collisions():
    sorted_x = sorted(ball_list, key=lambda i: i.p()[0])
    i = 0
    while i + 1 < len(sorted_x):
        if np.linalg.norm(sorted_x[i].p() - sorted_x[i + 1].p()) < 1.1 * (sorted_x[i].r + sorted_x[i + 1].r):
            new_v1 = vel_flip(unit_vector(sorted_x[i].p() - sorted_x[i + 1].p()), sorted_x[i].v(), sorted_x[i + 1].v())
            new_v2 = vel_flip(unit_vector(sorted_x[i + 1].p() - sorted_x[i].p()), sorted_x[i + 1].v(), sorted_x[i].v())
            Sim.velocity_array[sorted_x[i].no] = new_v1
            Sim.velocity_array[sorted_x[i + 1].no] = new_v2
            i += 2
        else:
            i += 1


def check_box_collisions():
    for i in ball_list:
        if (np.abs(i.p()[0]) > box_attributes[0] / 2 - i.r) and (np.abs(i.p()[1]) > box_attributes[1] / 2 - i.r):
            Sim.velocity_array[i.no] = vel_flip(np.array((0, -np.sign(i.p()[1]))), i.v(), -i.v())
            Sim.velocity_array[i.no] = vel_flip(np.array((-np.sign(i.p()[0]), 0)), i.v(), -i.v())
        elif np.abs(i.p()[0]) > box_attributes[0] / 2 - i.r and (i.p()[0] * i.v()[0]) > 0:
            Sim.velocity_array[i.no] = vel_flip(np.array((-np.sign(i.p()[0]), 0)), i.v(), -i.v())
        elif np.abs(i.p()[1]) > box_attributes[1] / 2 - i.r and (i.p()[1] * i.v()[1]) > 0:
            Sim.velocity_array[i.no] = vel_flip(np.array((0, -np.sign(i.p()[1]))), i.v(), -i.v())


def kernel(q, h=None):
    if h is None:
        h = Sim.smoothening_radius
    normalising_constant = 10 / (7 * math.pi)
    res = 0.0
    if q < 1:
        res = (normalising_constant / h ** 2) * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
    elif q < 2:
        res = (normalising_constant / h ** 2) * (0.25 * (2 - q) ** 3)
    return res


def kernel_gradient(q, direction, h=None):
    if h is None:
        h = Sim.smoothening_radius
    normalising_constant = 10 / (7 * math.pi)
    res = np.zeros(2, dtype=float)
    if q < 1:
        res = -direction * (normalising_constant / h ** 3) * (-3 * q + 2.25 * q ** 2)
    elif q < 2:
        res = -direction * (normalising_constant / h ** 3) * (-0.75 * (2 - q) ** 2)
    return res


def get_distance(particle1, particle2):
    if particle1.no == particle2.no:
        return 0
    distance = Sim.cached_distances[min(particle1.no, particle2.no)][
        max(particle1.no, particle2.no) - min(particle1.no, particle2.no) - 1]
    if distance != 0:
        return distance
    else:
        Sim.distance_count += 1
        distance = np.linalg.norm(particle2.p() - particle1.p())
        Sim.cached_distances[min(particle1.no, particle2.no)][max(particle1.no, particle2.no) - min(particle1.no,
                                                                                                    particle2.no) - 1] = distance
        return distance


def density_at(particle, chunk_population):
    density = 0
    for i in chunk_population:
        distance = get_distance(particle, i)
        density += i.m * kernel(distance / Sim.smoothening_radius)
    return density


def pressure_at(particle):
    return Sim.pressure_multiplier * (particle.density() - Sim.target_density)


def force_calc(particle, chunk_population):
    force = np.zeros(2, dtype=float)
    for j in chunk_population:
        if j == particle:
            continue
        if j.density() == 0:
            continue
        avg_pressure = (pressure_at(j) + pressure_at(particle)) / 2
        distance = get_distance(particle, j)
        if distance == 0: continue
        direction = (j.p() - particle.p()) / distance
        force += j.m * (avg_pressure / j.density()) * kernel_gradient(distance / Sim.smoothening_radius, direction)
    return -force


def cell_key(position):
    grid_w = box_attributes[0] // Sim.smoothening_radius
    grid_x = int((position[0] + box_attributes[0] / 2) / Sim.smoothening_radius)
    grid_y = int((position[1] + box_attributes[1] / 2) / Sim.smoothening_radius)
    return int(grid_y * grid_w + grid_x)


def generate_start_indices(ball_list_sorted):
    start_indices = {}
    if not ball_list_sorted:
        return start_indices
    current_key = cell_key(ball_list_sorted[0].p())
    start_indices[current_key] = 0
    for i in range(1, len(ball_list_sorted)):
        key = cell_key(ball_list_sorted[i].p())
        if key != current_key:
            start_indices[key] = i
            current_key = key
    return start_indices


def update_chunk_densities(i, j):
    chunk_population = []
    cell_population = []
    grid_w = box_attributes[0] // Sim.smoothening_radius
    for x in range(i - 1, i + 2):
        for y in range(j - 1, j + 2):
            key = y * grid_w + x
            start_index = Sim.start_indices.get(key, -1)
            if start_index != -1:
                for ball in ball_list[start_index:]:
                    if cell_key(ball.p()) != key:
                        break
                    chunk_population.append(ball)
                    if x == i and y == j:
                        cell_population.append(ball)


    for ball in cell_population:
        Sim.density_array[ball.no] = density_at(ball, chunk_population)


def update_chunk_forces(i, j):
    chunk_population = []
    cell_population = []
    grid_w = box_attributes[0] // Sim.smoothening_radius
    for x in range(i - 1, i + 2):
        for y in range(j - 1, j + 2):
            key = y * grid_w + x
            start_index = Sim.start_indices.get(key, -1)
            if start_index != -1:
                for ball in ball_list[start_index:]:
                    if cell_key(ball.p()) != key:
                        break
                    chunk_population.append(ball)
                    if x == i and y == j:
                        cell_population.append(ball)
    for ball in cell_population:
        Sim.acceleration_array[ball.no] += force_calc(ball, chunk_population)

class Profiler:
    def __init__(self):
        self.timings = {}
        self.start_time = 0

    def start(self):
        self.start_time = CPUTime.perf_counter()

    def stop(self, name):
        elapsed = (CPUTime.perf_counter() - self.start_time) * 1000  # to ms
        self.timings[name] = self.timings.get(name, elapsed) * 0.95 + elapsed * 0.05

    def display(self):
        total_time = 0
        print("\n--- Frame Performance (ms) ---")
        for name, timing in sorted(self.timings.items()):
            print(f"{name:<25}: {timing:.3f} ms")
            total_time+=timing
        print(f"FPS: {1000/total_time}")
        print(f"Total Time: {total_time}")
        print("-" * 30)

    def total_time(self):
        total_time = 0
        for name, timing in sorted(self.timings.items()):
            total_time+=timing
        return total_time

profiler = Profiler()


@win.event
def on_draw():
    profiler.start()
    win.clear()
    main_batch.draw()
    profiler.stop("Draw Call Time")


frame_count = 0
time = 0


def update(dt):
    global frame_count, ball_list, time
    if frame_count >= 100:
        print(profiler.total_time())
        sys.exit()

    profiler.start()
    Sim.distance_count = 0
    n = ball_count
    Sim.cached_distances = np.zeros((n, n))
    ball_list = sorted(ball_list, key=lambda i: cell_key(i.p()))
    Sim.start_indices = generate_start_indices(ball_list)
    profiler.stop("1. Sorting+Start indices")

    profiler.start()
    for j in range(int(box_attributes[1] / Sim.smoothening_radius)):
        for i in range(int(box_attributes[0] / Sim.smoothening_radius)):
            update_chunk_densities(i, j)
    profiler.stop("3. Density Loop Phase")
    profiler.start()
    Sim.acceleration_array.fill(0)
    for j in range(int(box_attributes[1] / Sim.smoothening_radius)):
        for i in range(int(box_attributes[0] / Sim.smoothening_radius)):
            update_chunk_forces(i, j)
    profiler.stop("Inner Force Calc")
    profiler.start()
    Sim.update_parameters()
    profiler.stop("5. Update Parameters")

    profiler.start()
    check_box_collisions()
    profiler.stop("6. Box Collisions")

    profiler.start()
    for i in ball_list:
        i.update_parameters()
    profiler.stop("7. Update Sprites")

    time += dt
    frame_count += 1



if __name__ == "__main__":
    pyglet.clock.schedule_interval(update, dt)
    pyglet.app.run()