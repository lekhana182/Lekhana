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
