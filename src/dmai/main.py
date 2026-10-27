import argparse
import logging
import os
import sys

from .com import AppConfig
from .core import DynamicClusterRouterPipeline


def action_runner():
    """Decorator to load configuration and print/log it upon action execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            config = AppConfig()
            logger = config.logger
            
            # --- Display loaded configuration upon startup ---
            print("\n" + "=" * 65)
            print("    ⚙️ [AppConfig] Loaded Configuration Overview")
            print("=" * 65)
            print(f" • Project Root               : {config.space_dir}")
            print(f" • Dataset Directory          : {config.dataset_dir}")
            print(f" • Log Directory              : {config.log_dir}")
            print(f" • Space Directory            : {config.space_dir}")
            print(f" • Similarity Threshold       : {getattr(config, 'similarity_threshold', 0.85)}")
            print(f" • Training Trigger Count     : {getattr(config, 'training_trigger_count', 3000)}")
            print(f" • Max Samples Per Class      : {getattr(config, 'max_samples_per_class', 300)}")
            print("=" * 65 + "\n")
            
            logger.info(
                f"Loaded Configs -> SimThresh: {getattr(config, 'similarity_threshold', 0.85)}, "
                f"TriggerCount: {getattr(config, 'training_trigger_count', 3000)}, "
                f"MaxPerClass: {getattr(config, 'max_samples_per_class', 300)}"
            )

            try:
                return func(config, logger, *args, **kwargs)
            except KeyboardInterrupt:
                msg = "[ABORT] Process interrupted by user."
                logger.warning(msg)
                print(f"\n\n{msg}\n")
        return wrapper
    return decorator


@action_runner()
def run_streaming_simulation(config, logger):
    """[1] Run full-scale streaming ingestion & clustering simulation using MNIST training data."""
    print("\n--- Starting Full-Scale Dynamic Cluster Router Simulation ---")
    logger.info("Starting full-scale streaming ingestion simulation based on app.toml configs.")
    pipeline = DynamicClusterRouterPipeline(config=config)

    from .dao import MNISTDataLoader
    loader = MNISTDataLoader(config=config)
    X_train, y_train = loader.load_train_data_by_classes(0, 9)
    pipeline.simulate_streaming_ingestion(X_train, y_train, num_samples=-1)
    print("\n✅ Full simulation completed successfully.")


@action_runner()
def run_full_evaluation(config, logger):
    """[2] Run full evaluation across 10,000 test samples (overall & per-class accuracy)."""
    print("\n--- Full Test Set Evaluation (10,000 Samples) ---")
    pipeline = DynamicClusterRouterPipeline(config=config)
    
    from .dao import MNISTDataLoader
    loader = MNISTDataLoader(config=config)
    X_test, y_test = loader.load_test_data()
    
    pipeline.evaluate_full_testset(X_test, y_test)


@action_runner()
def run_model_init(config, logger):
    """[9] Reset and remove all local classifiers, accumulated datasets, and state JSON files."""
    print("\n--- Resetting All Clusters, Datasets, and State JSON ---")
    space_dir = getattr(config, "space_dir", "space")
    import shutil
    
    for sub in ["expert_models", "expert_datasets"]:
        p = os.path.join(space_dir, sub)
        if os.path.exists(p):
            shutil.rmtree(p)
            print(f"Removed directory: {p}")
            
    state_file = os.path.join(space_dir, "clusters_state.json")
    if os.path.exists(state_file):
        os.remove(state_file)
        print(f"Removed state file: {state_file}")

    weights_path = os.path.join(space_dir, "feature_extractor_weights.pth")
    if os.path.exists(weights_path):
        os.remove(weights_path)
        print(f"Removed weights file: {weights_path}")
        
    print("\n✅ Initialization and reset completed successfully.")


def interactive_menu():
    """Interactive CLI Menu"""
    while True:
        print("\n" + "=" * 65)
        print("    Dynamic Vector Router & General Models System")
        print("=" * 65)
        print("    [1]  Simulate Streaming Ingestion & Clustering")
        print("    [2]  Run Full Evaluation on 10,000 Test Data")
        print("    ---------------------------------------------------------")
        print("    [9]  Init - Reset All Clusters & State")
        print("    [q]  Quit - Exit Application")
        print("=" * 65)

        try:
            choice = input("Enter choice: ").strip().lower()
        except KeyboardInterrupt:
            print("\nExiting. Thank you!")
            break

        if choice in ["q", "quit"]:
            print("Exiting. Thank you!")
            break
        elif choice in ["1"]:
            run_streaming_simulation()
        elif choice in ["2"]:
            run_full_evaluation()
        elif choice in ["9", "init"]:
            confirm = input("⚠️ All clusters, datasets, and state files will be deleted. Are you sure? (y/n): ").strip().lower()
            if confirm == "y":
                run_model_init()
            else:
                print("Initialization canceled.")
        else:
            print("[ERROR] Invalid choice. Please try again.")


def cli():
    parser = argparse.ArgumentParser(description="Dynamic Vector Router & General Models System")
    parser.add_argument("action", nargs="?", help="Action to execute")
    args = parser.parse_args()

    try:
        action = args.action.lower() if args.action else ""
        if action in ["1"]:
            run_streaming_simulation()
        elif action in ["2"]:
            run_full_evaluation()
        elif action in ["9", "init"]:
            run_model_init()
        else:
            interactive_menu()
    except KeyboardInterrupt:
        print("\n\n[ABORT] Program terminated.\n")
        sys.exit(0)


if __name__ == "__main__":
    cli()