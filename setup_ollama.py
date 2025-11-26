#!/usr/bin/env python3
"""
Setup script to verify Ollama installation and download required models
"""

import subprocess
import sys
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import requests

console = Console()

def check_ollama_installation():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            console.print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            console.print("❌ Ollama is installed but not responding properly")
            return False
    except FileNotFoundError:
        console.print("❌ Ollama is not installed")
        return False

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        response = requests.get('http://localhost:11434/api/version', timeout=5)
        if response.status_code == 200:
            console.print("✅ Ollama service is running")
            return True
        else:
            console.print("❌ Ollama service is not responding")
            return False
    except requests.exceptions.RequestException:
        console.print("❌ Cannot connect to Ollama service")
        return False

def download_model(model_name):
    """Download a model using Ollama"""
    console.print(f"📥 Downloading model: {model_name}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(description=f"Downloading {model_name}...", total=None)
        
        try:
            result = subprocess.run(
                ['ollama', 'pull', model_name],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                console.print(f"✅ Model {model_name} downloaded successfully")
                return True
            else:
                console.print(f"❌ Failed to download {model_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            console.print(f"❌ Download timeout for {model_name}")
            return False
        except Exception as e:
            console.print(f"❌ Error downloading {model_name}: {e}")
            return False

def list_available_models():
    """List locally available models"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            console.print("📋 Available local models:")
            console.print(result.stdout)
            return result.stdout.strip().split('\n')[1:]  # Skip header
        else:
            console.print("❌ Failed to list models")
            return []
    except Exception as e:
        console.print(f"❌ Error listing models: {e}")
        return []

def recommend_models():
    """Recommend models for synthetic data generation"""
    recommendations = {
        "granite4:latest": "IBM's latest granite model, excellent for data generation",
        "granite-code": "Optimized for structured data and code patterns",
        "mistral": "Fast and efficient alternative",
        "phi": "Lightweight option for basic generation"
    }
    
    console.print("\n🎯 Recommended models for synthetic data generation:")
    for model, description in recommendations.items():
        console.print(f"  • {model}: {description}")

def main():
    """Main setup function"""
    console.print("[bold blue]🔧 Ollama Setup for Synthetic Data Generation[/bold blue]\n")
    
    # Check installation
    if not check_ollama_installation():
        console.print("\n📖 Please install Ollama from: https://ollama.ai/")
        console.print("   For macOS: brew install ollama")
        console.print("   For Linux: curl https://ollama.ai/install.sh | sh")
        console.print("   For Windows: Download from https://ollama.ai/download")
        return False
    
    # Check service
    if not check_ollama_service():
        console.print("\n🚀 Please start the Ollama service:")
        console.print("   Run: ollama serve")
        console.print("   Or start the Ollama app")
        return False
    
    # List current models
    models = list_available_models()
    
    # Recommend models
    recommend_models()
    
    # Offer to download recommended model
    if not any('granite4:latest' in model for model in models):
        console.print("\n💡 Would you like to download granite4:latest (recommended)?")
        response = input("Download granite4:latest? (y/N): ").lower().strip()
        
        if response in ['y', 'yes']:
            if download_model('granite4:latest'):
                console.print("✅ Setup complete! You can now use the synthetic data generator.")
            else:
                console.print("❌ Setup incomplete. Please try downloading manually: ollama pull granite4:latest")
        else:
            console.print("ℹ️ You can download models later with: ollama pull <model-name>")
    else:
        console.print("✅ Setup complete! You have models available for synthetic data generation.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)