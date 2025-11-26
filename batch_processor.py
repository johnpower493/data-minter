#!/usr/bin/env python3
"""
Batch Processor for Multiple CSV Files

Processes multiple CSV files in batch to generate synthetic data
for entire data pipelines or database schemas.
"""

import os
import glob
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
from rich.console import Console
from rich.progress import Progress, TaskID
import typer
from synthetic_data_generator import SyntheticDataGenerator, GenerationConfig
import yaml
import json

console = Console()

class BatchProcessor:
    """Process multiple CSV files for synthetic data generation"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        
    def process_directory(self, input_dir: str, output_dir: str, pattern: str = "*.csv") -> Dict[str, Any]:
        """Process all CSV files in a directory"""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all CSV files
        csv_files = list(input_path.glob(pattern))
        
        if not csv_files:
            console.print(f"[yellow]No CSV files found in {input_dir}[/yellow]")
            return {"processed_files": [], "errors": []}
        
        console.print(f"[blue]Found {len(csv_files)} CSV files to process[/blue]")
        
        results = {"processed_files": [], "errors": []}
        
        with Progress() as progress:
            task = progress.add_task("[green]Processing files...", total=len(csv_files))
            
            for csv_file in csv_files:
                try:
                    result = self._process_single_file(csv_file, output_path)
                    results["processed_files"].append(result)
                    
                except Exception as e:
                    error_info = {"file": str(csv_file), "error": str(e)}
                    results["errors"].append(error_info)
                    console.print(f"[red]Error processing {csv_file.name}: {e}[/red]")
                
                progress.update(task, advance=1)
        
        return results
    
    def _process_single_file(self, csv_file: Path, output_dir: Path) -> Dict[str, Any]:
        """Process a single CSV file"""
        
        # Generate output filename
        output_file = output_dir / f"synthetic_{csv_file.name}"
        
        # Create generator
        generator = SyntheticDataGenerator(self.config)
        
        # Process file
        profiles = generator.analyze_csv(str(csv_file))
        synthetic_df = generator.generate_synthetic_data(profiles)
        
        # Save result
        if self.config.output_format.lower() == 'csv':
            synthetic_df.to_csv(output_file, index=False)
        elif self.config.output_format.lower() == 'json':
            output_file = output_file.with_suffix('.json')
            synthetic_df.to_json(output_file, orient='records', indent=2)
        elif self.config.output_format.lower() == 'parquet':
            output_file = output_file.with_suffix('.parquet')
            synthetic_df.to_parquet(output_file, index=False)
        
        return {
            "input_file": str(csv_file),
            "output_file": str(output_file),
            "rows_generated": len(synthetic_df),
            "columns_processed": len(profiles)
        }

def main():
    """CLI interface for batch processing"""
    app = typer.Typer()
    
    @app.command()
    def process_batch(
        input_dir: str = typer.Argument(..., help="Directory containing CSV files"),
        output_dir: str = typer.Argument(..., help="Output directory for synthetic data"),
        config_file: str = typer.Option(None, "--config", help="YAML configuration file"),
        pattern: str = typer.Option("*.csv", "--pattern", help="File pattern to match"),
        num_rows: int = typer.Option(1000, "--num-rows", help="Number of rows per file"),
        model: str = typer.Option("granite4:latest", "--model", help="Ollama model to use"),
    ):
        """Process multiple CSV files in batch"""
        
        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            config = GenerationConfig(**config_dict)
        else:
            config = GenerationConfig(num_rows=num_rows, ollama_model=model)
        
        # Create processor
        processor = BatchProcessor(config)
        
        # Process files
        results = processor.process_directory(input_dir, output_dir, pattern)
        
        # Display results
        console.print(f"\n[green]✅ Batch processing complete![/green]")
        console.print(f"Files processed: {len(results['processed_files'])}")
        console.print(f"Errors: {len(results['errors'])}")
        
        if results['errors']:
            console.print("\n[red]Errors encountered:[/red]")
            for error in results['errors']:
                console.print(f"  {error['file']}: {error['error']}")
    
    app()

if __name__ == "__main__":
    main()