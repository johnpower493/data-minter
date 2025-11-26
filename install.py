#!/usr/bin/env python3
"""
Installation and Setup Script for Synthetic Data Generator

Handles complete setup including dependency installation and Ollama configuration.
"""

import subprocess
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def install_dependencies():
    """Install Python dependencies"""
    console.print("[bold blue]📦 Installing Python dependencies...[/bold blue]")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        console.print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_project():
    """Setup project structure and permissions"""
    console.print("[bold blue]🔧 Setting up project structure...[/bold blue]")
    
    # Make scripts executable
    scripts = [
        "synthetic_data_generator.py",
        "setup_ollama.py", 
        "data_quality_validator.py",
        "batch_processor.py",
        "examples/run_examples.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            os.chmod(script, 0o755)
    
    # Create output directories
    output_dirs = ["output", "examples/output", "reports"]
    for dir_name in output_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    console.print("✅ Project structure setup complete")
    return True

def main():
    """Main installation process"""
    console.print("[bold green]🚀 Synthetic Data Generator Setup[/bold green]\n")
    
    success = True
    
    # Install dependencies
    if not install_dependencies():
        success = False
    
    # Setup project
    if not setup_project():
        success = False
    
    if success:
        console.print("\n[bold green]✅ Installation completed successfully![/bold green]")
        console.print("\nNext steps:")
        console.print("1. Run: python setup_ollama.py")
        console.print("2. Try: python examples/run_examples.py")
        console.print("3. Generate data: python synthetic_data_generator.py generate your_data.csv")
    else:
        console.print("\n[bold red]❌ Installation failed. Please check errors above.[/bold red]")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)