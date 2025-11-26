#!/usr/bin/env python3
"""
Example usage of the synthetic data generator
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data_generator import SyntheticDataGenerator, GenerationConfig
from rich.console import Console

console = Console()

def example_basic_generation():
    """Basic example of generating synthetic data"""
    
    console.print("[bold blue]Example 1: Basic Synthetic Data Generation[/bold blue]\n")
    
    # Configure generation
    config = GenerationConfig(
        num_rows=100,
        ollama_model="granite4:latest",
        output_format="csv",
        seed=42
    )
    
    # Create generator
    generator = SyntheticDataGenerator(config)
    
    # Analyze sample data
    csv_path = "examples/sample_customer_data.csv"
    profiles = generator.analyze_csv(csv_path)
    
    # Generate synthetic data
    synthetic_df = generator.generate_synthetic_data(profiles)
    
    # Save results
    output_path = "examples/synthetic_customer_data.csv"
    generator.save_data(synthetic_df, output_path)
    
    # Generate report
    report = generator.generate_report(profiles, synthetic_df)
    
    console.print(f"✅ Generated {len(synthetic_df)} rows")
    console.print(f"📁 Saved to: {output_path}")

def example_high_volume_generation():
    """Example of generating large datasets"""
    
    console.print("[bold blue]Example 2: High Volume Generation[/bold blue]\n")
    
    config = GenerationConfig(
        num_rows=10000,
        ollama_model="granite4:latest",
        output_format="parquet",  # More efficient for large datasets
        seed=123,
        anonymization_level="high"
    )
    
    generator = SyntheticDataGenerator(config)
    
    csv_path = "examples/sample_customer_data.csv"
    profiles = generator.analyze_csv(csv_path)
    
    synthetic_df = generator.generate_synthetic_data(profiles)
    
    output_path = "examples/large_synthetic_dataset.parquet"
    generator.save_data(synthetic_df, output_path)
    
    console.print(f"✅ Generated {len(synthetic_df)} rows")
    console.print(f"📁 Saved to: {output_path}")

def example_custom_configuration():
    """Example using custom configuration"""
    
    console.print("[bold blue]Example 3: Custom Configuration[/bold blue]\n")
    
    # Custom configuration for specific use case
    config = GenerationConfig(
        num_rows=500,
        preserve_relationships=True,
        anonymization_level="medium",
        output_format="json",
        ollama_model="granite4:latest",
        custom_patterns={
            "email_domains": ["testcompany.com", "example.org"],
            "phone_country_codes": ["+1"],
            "id_prefixes": ["TST"]
        }
    )
    
    generator = SyntheticDataGenerator(config)
    
    csv_path = "examples/sample_customer_data.csv"
    profiles = generator.analyze_csv(csv_path)
    
    synthetic_df = generator.generate_synthetic_data(profiles)
    
    output_path = "examples/custom_synthetic_data.json"
    generator.save_data(synthetic_df, output_path)
    
    console.print(f"✅ Generated {len(synthetic_df)} rows with custom patterns")
    console.print(f"📁 Saved to: {output_path}")

def run_all_examples():
    """Run all examples"""
    
    console.print("[bold green]🎯 Running Synthetic Data Generation Examples[/bold green]\n")
    
    try:
        example_basic_generation()
        console.print()
        
        example_custom_configuration()
        console.print()
        
        # Skip high volume example by default (can be slow)
        console.print("[dim]Skipping high volume example (uncomment to run)[/dim]")
        # example_high_volume_generation()
        
        console.print("[bold green]✅ All examples completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error running examples: {e}[/bold red]")
        console.print("\n💡 Make sure Ollama is running and you have the required model:")
        console.print("   ollama pull granite4:latest")

if __name__ == "__main__":
    run_all_examples()