#!/usr/bin/env python3
import os
import logging
import sys

def create_directory_structure():
    """Create the necessary directory structure for the project."""
    # Define the directories to create
    directories = [
        "core",
        "models",
        "utils",
        "config",
        "tests",
        "logs",
        "examples"
    ]
    
    # Get the project root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Initializing project structure in: {root_dir}")
    
    # Create each directory
    for directory in directories:
        dir_path = os.path.join(root_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {directory}/")
        else:
            print(f"Directory already exists: {directory}/")
    
    # Create __init__.py files in each directory for proper Python package structure
    for directory in directories:
        init_file = os.path.join(root_dir, directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                pass  # Create an empty file
            print(f"Created __init__.py in {directory}/")
    
    # Create a root __init__.py
    root_init = os.path.join(root_dir, "__init__.py")
    if not os.path.exists(root_init):
        with open(root_init, "w") as f:
            pass
        print("Created root __init__.py")
    
    # Create logs directory with .gitkeep
    logs_dir = os.path.join(root_dir, "logs")
    gitkeep_file = os.path.join(logs_dir, ".gitkeep")
    if not os.path.exists(gitkeep_file):
        with open(gitkeep_file, "w") as f:
            pass
        print("Created logs/.gitkeep")

def setup_logging():
    """Set up basic logging configuration."""
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    
    # Ensure logs directory exists
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Configure logging
    log_file = os.path.join(logs_dir, "init.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("Logging initialized")

def main():
    """Main function to initialize the project."""
    # Set up logging
    setup_logging()
    
    # Create directory structure
    create_directory_structure()
    
    logging.info("Project initialization complete")
    print("\nProject initialization complete!")
    print("You can now run 'pip install -r requirements.txt' to install dependencies")

if __name__ == "__main__":
    main() 