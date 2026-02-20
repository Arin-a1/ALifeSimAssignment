from parallel_hill_climber import ParallelHillClimber

if __name__ == "__main__":
    phc = ParallelHillClimber(config_path="config.yaml", pop_size=5)
    phc.evolve(generations=10)



