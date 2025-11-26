# Batch Processing Guide

The `batch_processor.py` utility allows you to process multiple CSV files at once, making it perfect for generating synthetic data for entire database schemas or data pipelines.

## 🎯 Overview

Instead of running the synthetic data generator on each file individually:
```bash
# Manual approach (tedious)
python synthetic_data_generator.py generate customers.csv --num-rows 1000
python synthetic_data_generator.py generate orders.csv --num-rows 1000  
python synthetic_data_generator.py generate products.csv --num-rows 1000
# ... repeat for each file
```

Use batch processing to handle all files at once:
```bash
# Batch approach (efficient)
python batch_processor.py process-batch input_data/ output_data/ --num-rows 1000
```

## 📂 Directory Structure

### Input Structure
```
input_data/
├── customers.csv
├── orders.csv
├── products.csv
├── reviews.csv
├── payments.csv
└── inventory.csv
```

### Output Structure
```
output_data/
├── synthetic_customers.csv
├── synthetic_orders.csv
├── synthetic_products.csv
├── synthetic_reviews.csv
├── synthetic_payments.csv
└── synthetic_inventory.csv
```

## 🖥️ Command Reference

### Basic Command
```bash
python batch_processor.py process-batch <input_dir> <output_dir> [options]
```

### Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_dir` | Directory containing CSV files | Required |
| `output_dir` | Directory for synthetic data output | Required |
| `--config-file` | YAML configuration file | None |
| `--pattern` | File pattern to match | `*.csv` |
| `--num-rows` | Number of rows per file | 1000 |
| `--model` | Ollama model to use | `granite4:latest` |

### Examples

#### Basic Batch Processing
```bash
# Process all CSV files in data/ directory
python batch_processor.py process-batch data/ output/
```

#### Custom Row Count
```bash
# Generate 5000 rows for each file
python batch_processor.py data/ output/ --num-rows 500 --model granite4:latest
```

#### Specific File Pattern
```bash
# Process only files ending with "_table.csv"
python batch_processor.py data/ output/ --pattern "*_table.csv"
```

#### Using Configuration File
```bash
# Use custom configuration for all files
python batch_processor.py data/ output/ \
  --config-file config/batch_config.yaml \
  --num-rows 10000
```

## ⚙️ Configuration

### Batch-Specific Configuration
Create a `batch_config.yaml` file for consistent settings across all files:

```yaml
# Batch processing configuration
num_rows: 5000
ollama_model: "granite4:latest"
output_format: "csv"
seed: 42

# Apply consistent precision rules to all files
preserve_decimal_places: false
max_decimal_places: 2

# Column-specific rules applied to all matching columns
column_precision_rules:
  price: 2
  cost: 2
  amount: 2
  quantity: 0
  count: 0

# Database compatibility for all files
database_compatibility: true
use_fixed_point: true
```

### Per-File Customization
While batch processing uses consistent settings, you can still customize per data type through the configuration file's column rules and patterns.

## 🚀 Cross-Platform Usage

### Windows PowerShell
```powershell
# Using the PowerShell script
.\run_production.ps1 -Command batch -InputFile "data\" -NumRows 5000

# Direct Python command
python batch_processor.py process-batch data\ output\ --num-rows 5000
```

### Linux/macOS
```bash
# Using Make
make batch DIR=data/ ROWS=5000

# Direct Python command  
python batch_processor.py process-batch data/ output/ --num-rows 5000
```

## 📊 Output and Reporting

### Console Output
```
🔄 Batch processing started...

Found 6 CSV files to process

Processing files... ████████████████████████████████████████ 100%

✅ Batch processing complete!
Files processed: 6
Errors: 0

Results:
  ✅ customers.csv → synthetic_customers.csv (5000 rows, 8 columns)
  ✅ orders.csv → synthetic_orders.csv (5000 rows, 12 columns)
  ✅ products.csv → synthetic_products.csv (5000 rows, 6 columns)
  ✅ reviews.csv → synthetic_reviews.csv (5000 rows, 5 columns)
  ✅ payments.csv → synthetic_payments.csv (5000 rows, 7 columns)
  ✅ inventory.csv → synthetic_inventory.csv (5000 rows, 4 columns)
```

### Error Handling
If some files fail to process:
```
⚠️ Batch processing complete with warnings!
Files processed: 5
Errors: 1

Results:
  ✅ customers.csv → synthetic_customers.csv (5000 rows)
  ✅ orders.csv → synthetic_orders.csv (5000 rows)
  ❌ corrupted_file.csv → Error: Invalid CSV format
  ✅ products.csv → synthetic_products.csv (5000 rows)
  ✅ reviews.csv → synthetic_reviews.csv (5000 rows)
  ✅ payments.csv → synthetic_payments.csv (5000 rows)

Errors encountered:
  corrupted_file.csv: Invalid CSV format
```

## 🎯 Use Cases

### 1. Database Schema Generation
Generate synthetic data for all tables in a database schema:
```bash
# Process entire database schema
python batch_processor.py process-batch schema_exports/ synthetic_db/ --num-rows 10000
```

### 2. ETL Pipeline Testing
Create test data for data pipeline validation:
```bash
# Generate pipeline test data
python batch_processor.py process-batch pipeline_samples/ test_data/ \
  --config-file pipeline_config.yaml
```

### 3. Development Environment Setup
Populate development databases across team:
```bash
# Create dev environment data
python batch_processor.py process-batch prod_samples/ dev_data/ \
  --num-rows 1000 \
  --model granite4:latest  # Latest granite model
```

### 4. CI/CD Integration
Automate synthetic data generation in build pipelines:
```bash
# In CI/CD script
python batch_processor.py process-batch test_schemas/ generated_test_data/ \
  --config-file ci_config.yaml \
  --num-rows 500
```

## 🔧 Advanced Features

### Large Dataset Processing
For processing many files or large files:
```yaml
# batch_large_config.yaml
num_rows: 50000
ollama_model: "granite4:latest"  # Latest granite model

advanced:
  llm_settings:
    batch_size: 100  # Larger batches for efficiency
    temperature: 0.2  # More consistent output
```

### Memory-Efficient Processing
For resource-constrained environments:
```yaml
# batch_memory_config.yaml
num_rows: 1000
ollama_model: "granite4:latest"  # Latest granite model

advanced:
  llm_settings:
    batch_size: 25   # Smaller batches
    temperature: 0.4
```

### Quality Validation Integration
Combine with quality validation:
```bash
# Process and validate all files
python batch_processor.py process-batch data/ output/ --config-file config.yaml

# Then validate each generated file
for file in output/synthetic_*.csv; do
  original="data/$(basename "$file" | sed 's/synthetic_//')"
  python data_quality_validator.py validate "$original" "$file"
done
```

## 🚨 Best Practices

### File Organization
- **Input directory**: Keep original CSV files organized by schema/domain
- **Output directory**: Use separate directory to avoid overwriting originals
- **Configuration**: Use version-controlled config files for reproducibility

### Performance Optimization
- **Model selection**: Use `granite4:latest` for best quality and latest features
- **Batch size**: Increase batch size for larger files, decrease for memory constraints
- **Row count**: Start with smaller row counts for initial testing

### Error Prevention
- **File validation**: Ensure all CSV files are properly formatted before batch processing
- **Disk space**: Check available disk space for output (synthetic files can be large)
- **Model availability**: Verify Ollama model is downloaded before starting large batches

### Security and Privacy
- **Data isolation**: Keep synthetic data separate from production data directories
- **Configuration security**: Don't commit sensitive configuration to version control
- **Local processing**: All processing stays local with Ollama - no data leaves your environment

## 🔗 Integration with Other Tools

### With Database Validator
```bash
# Generate and validate entire schema
python batch_processor.py process-batch schema/ synthetic_schema/ --num-rows 5000
python database_validator.py validate-csv synthetic_schema/synthetic_*.csv --schema-file schema.sql
```

### With Make/PowerShell Scripts
```bash
# Linux/Mac Makefile integration
make batch-generate DIR=data/ ROWS=10000
make batch-validate ORIGINAL_DIR=data/ SYNTHETIC_DIR=output/

# PowerShell integration  
.\run_production.ps1 -Command batch -InputFile data\ -NumRows 10000
```

### With Version Control
```bash
# Generate reproducible datasets
python batch_processor.py process-batch data/ output/ \
  --config-file config/v1.2.yaml \
  --seed 12345  # Reproducible results
```

---

## 📚 Related Documentation

- [Main README](README.md) - Getting started guide
- [Model Information](MODEL_INFO.md) - Choosing the right Granite model  
- [Database Validation](database_validator.py) - Validating generated data
- [Configuration Reference](config_template.yaml) - All available settings