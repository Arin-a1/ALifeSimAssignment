import copy
import numpy as np
import time
from robot import sample_robot
from simulator import Simulator
from utils import load_config

class ParallelHillClimber:
    def __init__(self, config_path, pop_size=10):
        self.config = load_config(config_path)
        self.pop_size = pop_size
        self.population = []
        self.fitness = np.zeros(pop_size)
        self.simulator = None
        self.mutation_rate = 0.3
        self.mutation_scale = 0.1
        self.elite_fraction = 0.2
        self.max_n_masses = 0
        self.max_n_springs = 0
        
        # For adaptive mutation
        self.stagnation_counter = 0
        self.best_fitness_history = []
        
        # Statistics tracking
        self.generation_times = []
        self.improvements = []
        
        # Initialize population with both body and random brain
        self.initialize_population()
    
    def initialize_population(self):
        """Create initial population with bodies and random neural networks"""
        print("Initializing population with bodies and brains...")
        
        # First create all robots to find max dimensions
        temp_population = []
        for _ in range(self.pop_size):
            robot = sample_robot()
            temp_population.append(robot)
            self.max_n_masses = max(self.max_n_masses, robot["n_masses"])
            self.max_n_springs = max(self.max_n_springs, robot["n_springs"])
        
        print(f"Max masses: {self.max_n_masses}, Max springs: {self.max_n_springs}")
        
        # Now create actual population with properly sized control params
        for robot in temp_population:
            # Add control parameters that match the MAX dimensions (for simulator)
            # The simulator expects arrays of size [max_n_masses * 4 + cpg_count, hidden_size]
            n_inputs = self.max_n_masses * 4 + self.config["simulator"]["nn_cpg_count"]
            n_hidden = self.config["simulator"]["nn_hidden_size"]
            n_outputs = self.max_n_springs
            
            robot["control_params"] = {
                "weights1": np.random.normal(0, 0.1, (n_inputs, n_hidden)).astype(np.float32),
                "weights2": np.random.normal(0, 0.1, (n_hidden, n_outputs)).astype(np.float32),
                "biases1": np.zeros(n_hidden, dtype=np.float32),
                "biases2": np.zeros(n_outputs, dtype=np.float32)
            }
            
            # Also store the actual dimensions for mutation
            robot["actual_n_masses"] = robot["n_masses"]
            robot["actual_n_springs"] = robot["n_springs"]
            
            self.population.append(robot)
    
    def setup_simulator(self):
        """Initialize the simulator once with appropriate dimensions"""
        self.config["simulator"]["n_sims"] = self.pop_size
        self.config["simulator"]["n_masses"] = self.max_n_masses
        self.config["simulator"]["n_springs"] = self.max_n_springs
        
        print(f"Setting up simulator with max masses: {self.max_n_masses}, max springs: {self.max_n_springs}")
        
        # Set needs_grad=False since we're doing evolutionary optimization
        self.simulator = Simulator(
            sim_config=self.config["simulator"],
            taichi_config=self.config["taichi"],
            seed=self.config["seed"],
            needs_grad=False
        )
    
    def evaluate_population(self):
        """Evaluate all robots in parallel and update self.fitness"""
        if self.simulator is None:
            self.setup_simulator()
        
        # Prepare robot data for batch simulation
        masses = [r["masses"] for r in self.population]
        springs = [r["springs"] for r in self.population]
        
        # Initialize simulator with current population
        self.simulator.initialize(masses, springs)
        
        # Load control parameters for each robot
        for i, robot in enumerate(self.population):
            if "control_params" in robot:
                # Create a copy of control params with the right shapes for this robot
                # The simulator expects full-sized arrays, but we only use the relevant parts
                self.simulator.set_control_params([i], [robot["control_params"]])
        
        # Run evaluation
        loss_values = self.evaluation_step()
        
        # Loss = com0 - comt, so negative loss means moved right
        self.fitness = -loss_values
        
        # Handle any NaN or inf values
        self.fitness = np.nan_to_num(self.fitness, nan=-1e6, posinf=-1e6, neginf=-1e6)
        
        print(f"  Fitness range: [{self.fitness.min():.4f}, {self.fitness.max():.4f}]")
        print(f"  Distance range: [{-loss_values.max():.4f}, {-loss_values.min():.4f}]")
    
    def evaluation_step(self):
        """Run a single evaluation without gradients"""
        self.simulator.reinitialize_robots()
        self.simulator.forward()
        self.simulator.compute_loss()
        return self.simulator.loss.to_numpy()
    
    def mutate_control_params(self, control_params):
        """Mutate the neural network weights - maintaining full size"""
        mutated = copy.deepcopy(control_params)
        
        if np.random.random() < self.mutation_rate:
            noise = np.random.normal(0, self.mutation_scale * 0.5, mutated["weights1"].shape)
            mutated["weights1"] += noise
        
        if np.random.random() < self.mutation_rate:
            noise = np.random.normal(0, self.mutation_scale * 0.5, mutated["weights2"].shape)
            mutated["weights2"] += noise
        
        if np.random.random() < self.mutation_rate:
            noise = np.random.normal(0, self.mutation_scale * 0.2, mutated["biases1"].shape)
            mutated["biases1"] += noise
        
        if np.random.random() < self.mutation_rate:
            noise = np.random.normal(0, self.mutation_scale * 0.2, mutated["biases2"].shape)
            mutated["biases2"] += noise
        
        return mutated
    
    def mutate_masses(self, masses, mutation_strength=None):
        """Mutate mass positions with small perturbations"""
        if mutation_strength is None:
            mutation_strength = self.mutation_scale
        
        mutated = masses.copy()
        
        if np.random.random() < self.mutation_rate:
            noise = np.random.normal(0, mutation_strength, mutated.shape)
            mutated += noise
            
            if mutated.shape[0] > 0:
                # Center horizontally
                x_center = np.mean(mutated[:, 0])
                mutated[:, 0] -= x_center
                
                # Ensure all masses are above ground
                min_y = np.min(mutated[:, 1])
                if min_y < 0.05:
                    mutated[:, 1] += (0.05 - min_y)
        
        return mutated
    
    def mutate_springs(self, springs, n_masses):
        """Mutate spring connections (topology)"""
        mutated = springs.copy()
        
        if np.random.random() < 0.05:  # 5% chance
            if np.random.random() < 0.5 and len(mutated) > 5:
                # Remove a spring
                idx = np.random.randint(len(mutated))
                mutated = np.delete(mutated, idx, axis=0)
            elif len(mutated) < n_masses * 2:
                # Add a new spring
                mass_a, mass_b = np.random.choice(n_masses, 2, replace=False)
                new_spring = np.array([mass_a, mass_b], dtype=np.int32)
                
                exists = False
                for spring in mutated:
                    if (spring[0] == mass_a and spring[1] == mass_b) or \
                       (spring[0] == mass_b and spring[1] == mass_a):
                        exists = True
                        break
                
                if not exists:
                    mutated = np.vstack([mutated, new_spring])
        
        return mutated
    
    def mutate(self, robot):
        """Create a mutated child with both body and brain mutations"""
        child = copy.deepcopy(robot)
        
        if np.random.random() < 0.3:  # 30% chance to mutate body
            child["masses"] = self.mutate_masses(robot["masses"])
            child["springs"] = self.mutate_springs(robot["springs"], robot["n_masses"])
            child["n_springs"] = len(child["springs"])
            child["actual_n_springs"] = child["n_springs"]
        
        if "control_params" in child:
            child["control_params"] = self.mutate_control_params(robot["control_params"])
        
        return child
    
    def adapt_mutation_rate(self, generation, max_generations):
        """Adapt mutation rate based on progress"""
        if len(self.best_fitness_history) > 3:
            recent_improvement = self.best_fitness_history[-1] - self.best_fitness_history[-3]
            
            if abs(recent_improvement) < 0.01:  # Stagnation
                self.stagnation_counter += 1
                self.mutation_rate = min(0.6, 0.3 + self.stagnation_counter * 0.1)
                self.mutation_scale = min(0.2, 0.1 + self.stagnation_counter * 0.03)
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)
                progress = generation / max_generations
                self.mutation_rate = max(0.1, 0.3 * (1 - progress * 0.5))
                self.mutation_scale = max(0.05, 0.1 * (1 - progress * 0.3))
    
    def save_robot_for_visualizer(self, robot, filename="best_robot.npy"):
        """Save robot with control parameters for visualization"""
        robot_copy = copy.deepcopy(robot)
        robot_copy["max_n_masses"] = self.max_n_masses
        robot_copy["max_n_springs"] = self.max_n_springs
        
        np.save(filename, robot_copy)
        print(f"  Saved robot with brain to {filename}")
    
    def evolve(self, generations=20):
        """Run the parallel hill climber"""
        print(f"\n{'='*50}")
        print(f"Starting evolution with population size: {self.pop_size}")
        print(f"Generations: {generations}")
        print(f"{'='*50}\n")
        
        # Initial evaluation
        print("Evaluating initial population...")
        start_time = time.time()
        self.evaluate_population()
        init_time = time.time() - start_time
        
        best_idx = np.argmax(self.fitness)
        best_fit = self.fitness[best_idx]
        self.best_fitness_history.append(best_fit)
        
        loss_values = self.simulator.loss.to_numpy()
        print(f"Initial best fitness: {best_fit:.4f} (moved {-loss_values[best_idx]:.4f} units right)")
        print(f"Initial evaluation took {init_time:.2f}s")
        self.save_robot_for_visualizer(self.population[best_idx], "best_robot_gen_0.npy")
        
        # Evolution loop
        for g in range(generations):
            gen_start = time.time()
            prev_best = best_fit
            
            # Adapt mutation rate
            self.adapt_mutation_rate(g, generations)
            
            # Create children with elite preservation
            n_elite = max(1, int(self.pop_size * self.elite_fraction))
            elite_indices = np.argsort(self.fitness)[-n_elite:]
            
            children = []
            for i in range(self.pop_size):
                if i in elite_indices:
                    children.append(copy.deepcopy(self.population[i]))
                else:
                    # Mutate a random parent
                    parent_idx = np.random.choice(self.pop_size)
                    child = self.mutate(self.population[parent_idx])
                    children.append(child)
            
            # Store old population
            old_population = self.population
            old_fitness = self.fitness.copy()
            
            # Evaluate children
            self.population = children
            self.evaluate_population()
            
            # Selection: keep better individual
            for i in range(self.pop_size):
                if old_fitness[i] > self.fitness[i]:
                    self.population[i] = old_population[i]
                    self.fitness[i] = old_fitness[i]
            
            # Track best fitness
            best_idx = np.argmax(self.fitness)
            best_fit = self.fitness[best_idx]
            self.best_fitness_history.append(best_fit)
            
            # Calculate statistics
            gen_time = time.time() - gen_start
            self.generation_times.append(gen_time)
            improvement = best_fit - prev_best
            
            # Get actual distance moved for the best robot
            loss_values = self.simulator.loss.to_numpy()
            best_distance = -loss_values[best_idx]
            
            print(f"\nGeneration {g+1}/{generations} completed in {gen_time:.2f}s:")
            print(f"  Best fitness: {best_fit:.4f} (moved {best_distance:.4f} units right)")
            print(f"  Improvement: {improvement:+.4f}")
            print(f"  Mean fitness: {np.mean(self.fitness):.4f} ± {np.std(self.fitness):.4f}")
            print(f"  Mutation rate: {self.mutation_rate:.3f}")
            
            # Save best robot periodically
            if (g + 1) % 5 == 0 or g == generations - 1:
                self.save_robot_for_visualizer(
                    self.population[best_idx], 
                    f"best_robot_gen_{g+1}.npy"
                )
        
        # Final summary
        print(f"\n{'='*50}")
        print("Evolution Complete!")
        print(f"{'='*50}")
        final_loss = self.simulator.loss.to_numpy()[best_idx]
        print(f"Final best robot moved {-final_loss:.4f} units to the RIGHT")
        print(f"Initial best moved {-self.best_fitness_history[0]:.4f} units")
        print(f"Total improvement: {best_fit - self.best_fitness_history[0]:.4f}")
        
        # Save final best robot
        self.save_robot_for_visualizer(self.population[best_idx], "best_robot_final.npy")
        
        return self.population[best_idx], best_fit


if __name__ == "__main__":
    CONFIG_PATH = "config.yaml"
    POP_SIZE = 10
    GENERATIONS = 20
    
    print("Starting Parallel Hill Climber with Body+Brain Evolution...")
    print(f"Config: {CONFIG_PATH}")
    print(f"Population size: {POP_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print("-" * 50)
    
    phc = ParallelHillClimber(config_path=CONFIG_PATH, pop_size=POP_SIZE)
    best_robot, best_fitness = phc.evolve(generations=GENERATIONS)
    
    print(f"\nBest robot achieved fitness: {best_fitness:.4f}")
    print("\nTo visualize the best robot with its learned behavior:")
    print(f"python visualizer.py --input best_robot_final.npy --config {CONFIG_PATH}")