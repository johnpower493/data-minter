# Synthetic Data Generation Tool 🎯

A powerful Python tool for generating high-quality synthetic data using local LLMs via Ollama. Perfect for DEV/UAT testing without relying on production data, ensuring privacy compliance and data security.

## 🚀 Features

- **Intelligent Data Type Inference**: Automatically detects column types (numeric, categorical, dates, emails, phones, etc.)
- **Local LLM Integration**: Uses Ollama for privacy-preserving synthetic data generation
- **Statistical Preservation**: Maintains original data distributions and relationships
- **Multiple Output Formats**: CSV, JSON, and Parquet support
- **Configurable Privacy Levels**: Low, medium, and high anonymization options
- **Rich CLI Interface**: Beautiful command-line interface with progress bars and tables
- **Extensible Architecture**: Easy to add custom data generators and patterns

## 📋 Prerequisites

1. **Python 3.8+**
2. **Ollama**: Local LLM runtime
   - Install from: https://ollama.ai/
   - macOS: `brew install ollama`
   - Linux: `curl https://ollama.ai/install.sh | sh`
   - Windows: Download from https://ollama.ai/download

**About IBM Granite Models**: This tool uses IBM's open-source Granite models, which are specifically designed for enterprise use cases with excellent performance on structured data tasks. Granite models offer strong privacy guarantees and are optimized for data generation workflows.

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Ollama:**
   ```bash
   python setup_ollama.py
   ```
   
   This will:
   - Verify Ollama installation
   - Check if service is running
   - Download recommended models
   - Provide setup guidance

## 🎯 Quick Start

### 1. Analyze Your Data
```bash
python synthetic_data_generator.py analyze your_data.csv
```

This will show detected data types and statistics for each column.

### 2. Generate Synthetic Data
```bash
python synthetic_data_generator.py generate your_data.csv
```

Basic generation with defaults (1000 rows, CSV output).

### 3. Advanced Usage
```bash
python synthetic_data_generator.py generate your_data.csv \
  --num-rows 5000 \
  --model granite4:latest \
  --format parquet \
  --output-path synthetic_data.parquet \
  --seed 42
```
### Single Line (Powershell)
```
python synthetic_data_generator.py generate examples\staff.csv --num-rows 500 --model granite4 --format csv --output-path examples\staff_dummy_data.csv --seed 42
```

### 4. Using Configuration File
```bash
python synthetic_data_generator.py generate your_data.csv \
  --config-file config_template.yaml
```

## 📁 Project Structure

```
├── synthetic_data_generator.py    # Main application
├── setup_ollama.py               # Ollama setup utility
├── requirements.txt               # Python dependencies
├── config_template.yaml          # Configuration template
├── examples/
│   ├── sample_customer_data.csv   # Example input data
│   └── run_examples.py           # Example usage scripts
└── README.md                     # This file
```

## 🔧 Configuration Options

### Basic Configuration
```yaml
num_rows: 1000
preserve_relationships: true
anonymization_level: "medium"  # low, medium, high
output_format: "csv"           # csv, json, parquet
seed: 42                       # for reproducible results
ollama_model: "granite4:latest"  # Latest granite model with best features
```

### Advanced Configuration
```yaml
advanced:
  preserve_correlations: true
  quality_controls:
    min_uniqueness_ratio: 0.8
    max_null_deviation: 5.0
  llm_settings:
    batch_size: 50
    temperature: 0.7
  privacy:
    anonymize_names: true
    scramble_ids: true
    generalize_locations: true
```

## 🎨 Supported Data Types

| Type | Description | Generation Strategy |
|------|-------------|-------------------|
| **Numeric** | Numbers, integers, floats | Statistical distribution matching |
| **Categorical** | Limited set of values | Frequency-based sampling |
| **Date/Time** | Temporal data | Range-based generation |
| **Email** | Email addresses | Pattern-based with custom domains |
| **Phone** | Phone numbers | Format-preserving generation |
| **Names** | Person names | LLM-generated realistic names |
| **Addresses** | Location data | LLM-generated addresses |
| **IDs** | Unique identifiers | Pattern-based with prefixes |
| **Currency** | Money values | Log-normal distribution |
| **Text** | Free-form text | LLM-generated content |

## 💡 Examples

### Run Built-in Examples
```bash
cd examples
python run_examples.py
```

### Python API Usage
```python
from synthetic_data_generator import SyntheticDataGenerator, GenerationConfig

# Configure generation
config = GenerationConfig(
    num_rows=1000,
    ollama_model="granite4:latest",
    anonymization_level="medium"
)

# Create generator and analyze data
generator = SyntheticDataGenerator(config)
profiles = generator.analyze_csv("your_data.csv")

# Generate synthetic data
synthetic_df = generator.generate_synthetic_data(profiles)

# Save results
generator.save_data(synthetic_df, "synthetic_output.csv")
```

## 🔒 Privacy & Security Features

- **Local Processing**: All data stays on your machine
- **No External APIs**: Uses local Ollama models only
- **Configurable Anonymization**: Multiple privacy levels
- **Pattern Scrambling**: Maintains format while changing values
- **Relationship Preservation**: Keeps statistical correlations

## 🚀 Use Cases

- **Development Testing**: Generate realistic test data for development
- **UAT Environment**: Populate staging environments safely
- **Data Science**: Create datasets for algorithm testing
- **Training Data**: Generate data for ML model training
- **Compliance Testing**: Test systems with privacy-safe data
- **Performance Testing**: Create large datasets for load testing

## 📊 Quality Metrics

The tool provides quality metrics comparing original vs synthetic data:
- Uniqueness ratios
- Statistical distribution preservation
- Null value consistency
- Pattern matching accuracy

## 🛠️ Troubleshooting

### Common Issues

1. **Ollama not found**: Install Ollama and ensure it's in your PATH
2. **Model not available**: Run `ollama pull granite4:latest` to download the model
3. **Service not running**: Start Ollama with `ollama serve`
4. **Generation slow**: Try smaller models like `granite4:1b` or `phi`
5. **Memory issues**: Reduce batch size in configuration

### Getting Help

1. Check if Ollama is running: `ollama list`
2. Verify model availability: `ollama pull granite4:latest`
3. Test basic generation with small row count
4. Check logs for detailed error messages

## 🤝 Contributing

This tool is designed to be extensible. You can:

- Add new data type detectors
- Implement custom generation strategies
- Add support for new output formats
- Improve LLM prompts for better generation

## 📄 License

Open source - feel free to use, modify, and distribute.

## 🔮 Roadmap

- [ ] Database connectivity (PostgreSQL, MySQL)
- [ ] Incremental data generation
- [ ] Advanced relationship modeling
- [ ] Custom data validation rules
- [ ] Integration with data catalogs
- [ ] Differential privacy options
- [ ] Real-time streaming generation

---

**Happy synthetic data generation! 🎉**