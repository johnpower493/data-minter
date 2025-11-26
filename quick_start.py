#!/usr/bin/env python3
"""
Quick Start Script for Synthetic Data Generator

Interactive setup and first-time usage guide.
"""

import os
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table

console = Console()

def welcome_banner():
    """Display welcome banner"""
    banner = """
🎯 Synthetic Data Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate privacy-safe synthetic data using local LLMs via Ollama!
Perfect for DEV/UAT testing without production data concerns.

Features:
✅ Intelligent data type inference
✅ Local LLM integration (no external APIs)
✅ Statistical property preservation
✅ Multiple output formats (CSV, JSON, Parquet)
✅ Quality validation and reporting
✅ Batch processing capabilities
    """
    
    console.print(Panel(banner, border_style="blue"))

def check_requirements():
    """Check if basic requirements are met"""
    console.print("[bold blue]🔍 Checking system requirements...[/bold blue]\n")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    else:
        console.print("✅ Python version OK")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        issues.append("requirements.txt not found")
    else:
        console.print("✅ Requirements file found")
    
    # Check if Ollama is installed
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, check=True)
        console.print("✅ Ollama is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("Ollama not installed - install from https://ollama.ai/")
    
    return issues

def guided_setup():
    """Guide user through setup process"""
    console.print("\n[bold blue]🚀 Let's set up your synthetic data generator![/bold blue]\n")
    
    # Check requirements
    issues = check_requirements()
    
    if issues:
        console.print("[red]⚠️  Setup issues found:[/red]")
        for issue in issues:
            console.print(f"   • {issue}")
        
        if not Confirm.ask("\nContinue anyway?"):
            return False
    
    # Install dependencies
    if Confirm.ask("\n📦 Install Python dependencies?"):
        console.print("Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            console.print("✅ Dependencies installed")
        except subprocess.CalledProcessError:
            console.print("❌ Failed to install dependencies")
            return False
    
    # Setup Ollama
    if Confirm.ask("\n🤖 Setup Ollama and download models?"):
        try:
            subprocess.check_call([sys.executable, "setup_ollama.py"])
        except subprocess.CalledProcessError:
            console.print("⚠️  Ollama setup had issues - you may need to set it up manually")
    
    return True

def quick_demo():
    """Run a quick demonstration"""
    console.print("\n[bold blue]🎬 Quick Demo[/bold blue]\n")
    
    # Check if example data exists
    example_file = "examples/sample_customer_data.csv"
    if not Path(example_file).exists():
        console.print(f"❌ Example file not found: {example_file}")
        return
    
    demo_choice = Prompt.ask(
        "What would you like to try?",
        choices=["analyze", "generate", "both", "skip"],
        default="both"
    )
    
    if demo_choice in ["analyze", "both"]:
        console.print("\n📊 Analyzing sample data...")
        try:
            subprocess.run([
                sys.executable, "synthetic_data_generator.py", 
                "analyze", example_file
            ])
        except subprocess.CalledProcessError:
            console.print("❌ Analysis failed")
    
    if demo_choice in ["generate", "both"]:
        console.print("\n🎯 Generating synthetic data...")
        num_rows = Prompt.ask("How many rows to generate?", default="100")
        
        try:
            subprocess.run([
                sys.executable, "synthetic_data_generator.py",
                "generate", example_file,
                "--num-rows", num_rows,
                "--output-path", "quick_demo_output.csv"
            ])
            
            if Path("quick_demo_output.csv").exists():
                console.print("✅ Demo data generated: quick_demo_output.csv")
            
        except subprocess.CalledProcessError:
            console.print("❌ Generation failed")

def show_next_steps():
    """Show next steps after setup"""
    
    table = Table(title="What's Next? 🚀")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")
    
    table.add_row(
        "python synthetic_data_generator.py analyze data.csv",
        "Analyze your CSV file and see detected data types"
    )
    table.add_row(
        "python synthetic_data_generator.py generate data.csv",
        "Generate synthetic data from your CSV"
    )
    table.add_row(
        "python examples/run_examples.py",
        "Run comprehensive examples"
    )
    table.add_row(
        "python data_quality_validator.py validate orig.csv synth.csv",
        "Validate quality of generated data"
    )
    table.add_row(
        "python batch_processor.py process-batch input/ output/",
        "Process multiple files in batch"
    )
    
    console.print("\n")
    console.print(table)
    
    console.print("\n[bold green]📚 For more information:[/bold green]")
    console.print("• Read README.md for detailed documentation")
    console.print("• Check config_template.yaml for configuration options")
    console.print("• Run 'make help' for available make commands")

def main():
    """Main quick start function"""
    welcome_banner()
    
    if not guided_setup():
        console.print("\n[red]Setup incomplete. Please resolve issues and try again.[/red]")
        return False
    
    console.print("\n[green]✅ Setup complete![/green]")
    
    if Confirm.ask("\n🎬 Run a quick demo?"):
        quick_demo()
    
    show_next_steps()
    
    console.print("\n[bold green]🎉 You're all set! Happy synthetic data generation![/bold green]")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)