import numpy as np
import random

class Collider:
    def __init__(self, collision_distance=1.0):
        self.collision_distance = collision_distance

    def detect_collisions(self, particles):
        collisions = []
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                distance = np.linalg.norm(particles[i].position - particles[j].position)
                if distance <= self.collision_distance:
                    collisions.append((particles[i], particles[j]))
        return collisions

    def resolve_collision(self, particle1, particle2):
        total_mass = particle1.mass + particle2.mass
        velocity = (particle1.mass * particle1.velocity + particle2.mass * particle2.velocity) / total_mass
        position = (particle1.position + particle2.position) / 2

        if particle1.energy + particle2.energy > 1e6:  # Fusion threshold
            new_particle = Particle(position, velocity, total_mass, "new_particle")
            return [new_particle]
        else:  # Stochastic splitting
            if random.random() < 0.5:
                return particle1.detect_decay() + particle2.detect_decay()
            else:
                return [particle1, particle2]
