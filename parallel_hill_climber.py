import copy
import numpy as np
from robot import sample_robot
from simulator import Simulator
from utils import load_config

class ParallelHillClimber:

    def __init__(self, config_path, pop_size):
        self.config = load_config(config_path)
        self.pop_size = pop_size
        self.population = [sample_robot() for _ in range(pop_size)]
        self.fitness = np.zeros(pop_size)

    def evaluate_population(self):
        """Evaluate all robots in parallel and update self.fitness"""
        max_masses = max(r["n_masses"] for r in self.population)
        max_springs = max(r["n_springs"] for r in self.population)

        self.config["simulator"]["n_sims"] = self.pop_size
        self.config["simulator"]["n_masses"] = max_masses
        self.config["simulator"]["n_springs"] = max_springs

        simulator = Simulator(
            sim_config=self.config["simulator"],
            taichi_config=self.config["taichi"],
            seed=self.config["seed"],
            needs_grad=True
        )

        masses = [r["masses"] for r in self.population]
        springs = [r["springs"] for r in self.population]

        simulator.initialize(masses, springs)

        fitness_history = simulator.train()
        self.fitness = fitness_history[:, -1]

    def mutate(self, robot):
        """Create a mutated child of the given robot"""
        child = sample_robot(p=0.3)
        return child

    def save_robot_for_visualizer(self, robot, filename="best_robot.npy"):
        """Trim the robot and save only what's needed for visualization"""
        vis_robot = {
            "n_masses": robot["n_masses"],
            "n_springs": robot["n_springs"],
            "masses": robot["masses"],
            "springs": robot["springs"]
        }

        if "control_params" in robot:
            vis_robot["control_params"] = robot["control_params"]

        np.save(filename, vis_robot)
        print(f"Saved visualizer-friendly robot to {filename}")

    def evolve(self, generations):
        """Run the parallel hill climber"""
        self.evaluate_population()

        for g in range(generations):
            print(f"\nGeneration {g}")

            children = [self.mutate(r) for r in self.population]

            old_population = self.population
            old_fitness = self.fitness.copy()

            self.population = children
            self.evaluate_population()

            for i in range(self.pop_size):
                if old_fitness[i] > self.fitness[i]:
                    self.population[i] = old_population[i]
                    self.fitness[i] = old_fitness[i]

            best_idx = np.argmax(self.fitness)
            best_fit = self.fitness[best_idx]
            print(f"Best fitness: {best_fit}")
            print("Best fitness so far:", np.max(self.fitness))


            self.save_robot_for_visualizer(self.population[best_idx], filename="best_robot.npy")


            


if __name__ == "__main__":
    phc = ParallelHillClimber(config_path="config.yaml", pop_size=5)
    phc.evolve(generations=15)
