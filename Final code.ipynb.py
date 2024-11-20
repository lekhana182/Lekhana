import random
import numpy as np

class Particle:
    def __init__(self, position, velocity, mass, particle_type):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.type = particle_type
        self.energy = self.calculate_kinetic_energy()

    def calculate_kinetic_energy(self):
        speed_squared = np.sum(self.velocity**2)
        return 0.5 * self.mass * speed_squared
        

    def update_position(self, time_step):
        self.position += self.velocity * time_step

    def detect_decay(self):
        decay_probability = 0.05  # Example decay chance
        if random.random() < decay_probability:
            return self.simulate_decay()
        return [self]

    def simulate_decay(self):
        if self.type == "proton":
            return [
                Particle(self.position, [0, 0, 0], 0.511, "electron"),
                Particle(self.position, [0, 0, 0], 0.0, "neutrino")
            ]
        elif self.type == "muon":
            return [
                Particle(self.position, [0, 0, 0], 0.105, "electron"),
                Particle(self.position, [0, 0, 0], 0.0, "neutrino")
            ]
        return [self]

    def __repr__(self):
        return f"Particle(type={self.type}, position={self.position.tolist()}, velocity={self.velocity.tolist()}, mass={self.mass}, energy={self.energy})"
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
class Simulation:
    def __init__(self, num_particles=100, time_step=0.01):
        self.time_step = time_step
        self.particles = self.initialize_particles(num_particles)
        self.collider = Collider()

    def initialize_particles(self, num_particles):
        particles = []
        for _ in range(num_particles):
            position = np.random.rand(3) * 100
            velocity = np.random.randn(3) * 20
            mass = random.uniform(1, 10)
            particle_type = random.choice(["proton", "electron", "muon"])
            particles.append(Particle(position, velocity, mass, particle_type))
        return particles

    def run(self, steps=1000):
        for _ in range(steps):
            for particle in self.particles:
                particle.update_position(self.time_step)
            collisions = self.collider.detect_collisions(self.particles)
            for p1, p2 in collisions:
                result = self.collider.resolve_collision(p1, p2)
                self.particles.remove(p1)
                self.particles.remove(p2)
                self.particles.extend(result)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def display_particles(self, particles):
        self.ax.clear()
        colors = {"proton": "red", "electron": "blue", "muon": "green"}
        for particle in particles:
            self.ax.scatter(*particle.position, color=colors.get(particle.type, "black"))
        plt.draw()
        plt.pause(0.10)
import csv
import json

class DataLogger:
    def __init__(self, filename="simulation_data"):
        self.filename = filename
        self.data = []

    def log_event(self, event_type, details):
        self.data.append({"event": event_type, "details": details})

    def save_to_csv(self):
        with open(f"{self.filename}.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["event", "details"])
            writer.writeheader()
            writer.writerows(self.data)

    def save_to_json(self):
        with open(f"{self.filename}.json", "w") as file:
            json.dump(self.data, file, indent=4)
if __name__ == "__main__":
    # Initialize components
    sim = Simulation(num_particles=100)  # Smaller number of particles for clarity
    visualizer = Visualizer()
    logger = DataLogger()

    steps = 75  # Number of simulation steps
    for step in range(steps):
        print(f"\nStep {step + 1}:")

        # Run one simulation step
        sim.run(steps=1)

        # Display particle states
        print(f"Particles ({len(sim.particles)} total):")
        for particle in sim.particles:
            print(particle)

        # Log particle count and any significant events
        logger.log_event("step", f"Step {step + 1}: {len(sim.particles)} particles")

        # Visualize particles
        visualizer.display_particles(sim.particles)

        # Detect and print decay or collision events
        collisions = sim.collider.detect_collisions(sim.particles)
        if collisions:
            print(f"Collisions detected: {len(collisions)}")
            for p1, p2 in collisions:
                print(f" - {p1.type} collided with {p2.type}")
                result = sim.collider.resolve_collision(p1, p2)
                logger.log_event("collision", {"particle1": p1.type, "particle2": p2.type, "result": [r.type for r in result]})
        else:
            print("No collisions this step.")

    # Save logs after the simulation
    logger.save_to_csv()
    print("\nSimulation complete. Data saved to simulation_data.csv.")