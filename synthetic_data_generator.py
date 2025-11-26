#!/usr/bin/env python3
"""
Synthetic Data Generation Tool using Local LLMs via Ollama

This tool analyzes CSV files and generates synthetic data that maintains
statistical properties while ensuring privacy and compliance.
"""

import pandas as pd
import numpy as np
import ollama
import json
import re
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from decimal import Decimal, getcontext, ROUND_HALF_UP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

class DataType(Enum):
    """Supported data types for synthetic generation"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATE = "date"
    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    ID = "id"
    BOOLEAN = "boolean"
    ADDRESS = "address"
    NAME = "name"
    CURRENCY = "currency"

@dataclass
class ColumnProfile:
    """Profile of a column including type and statistics"""
    name: str
    data_type: DataType
    sample_values: List[Any]
    unique_count: int
    null_percentage: float
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    pattern: Optional[str] = None
    categories: Optional[List[str]] = None
    date_range: Optional[tuple] = None
    decimal_places: Optional[int] = None

@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation"""
    num_rows: int = 1000
    preserve_relationships: bool = True
    anonymization_level: str = "medium"  # low, medium, high
    output_format: str = "csv"  # csv, json, parquet
    seed: Optional[int] = None
    ollama_model: str = "granite4:latest"
    custom_patterns: Dict[str, str] = None
    max_decimal_places: int = 2  # Maximum decimal places for float values
    preserve_decimal_places: bool = True  # Whether to preserve original decimal precision
    use_fixed_point: bool = True  # Use fixed-point decimals for financial data
    column_precision_rules: Optional[Dict[str, int]] = None  # Column-specific precision rules
    database_compatibility: bool = True  # Enable database compatibility checks

class DataTypeInferrer:
    """Infers data types from CSV columns using pattern matching and LLM assistance"""
    
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.phone_pattern = re.compile(r'^[\+]?[1-9]?[\d\s\-\(\)\.]{7,15}$')
        self.id_pattern = re.compile(r'^[A-Z0-9\-_]{8,}$|^\d{6,}$')
        self.generator = None  # Will be set by SyntheticDataGenerator
        
    def infer_column_type(self, series: pd.Series, column_name: str) -> ColumnProfile:
        """Infer the data type and create a profile for a column"""
        
        # Basic statistics
        non_null_values = series.dropna()
        sample_values = non_null_values.head(10).tolist()
        unique_count = series.nunique()
        null_percentage = (series.isnull().sum() / len(series)) * 100
        
        # Initial type detection
        data_type = self._detect_primary_type(non_null_values, column_name)
        
        # Create base profile
        profile = ColumnProfile(
            name=column_name,
            data_type=data_type,
            sample_values=sample_values,
            unique_count=unique_count,
            null_percentage=null_percentage
        )
        
        # Add type-specific statistics
        self._add_type_specific_stats(profile, non_null_values)
        
        return profile
    
    def _detect_primary_type(self, series: pd.Series, column_name: str) -> DataType:
        """Detect the primary data type of a series"""
        
        if len(series) == 0:
            return DataType.TEXT
        
        # Use column name hints first for better detection
        column_lower = column_name.lower()
        
        # Check for date/time columns by name first (highest priority)
        if any(hint in column_lower for hint in ['date', 'time', 'timestamp', 'created', 'updated', 'modified', 'last_update']):
            # Try to parse as datetime to confirm
            try:
                sample_data = series.dropna().head(5)
                if len(sample_data) > 0:
                    pd.to_datetime(sample_data, errors='raise')
                    return DataType.DATE
            except:
                # If parsing fails, check if it looks like time data
                sample_str = str(series.dropna().iloc[0]) if len(series.dropna()) > 0 else ""
                if re.match(r'\d{2}:\d{2}:\d{2}', sample_str) or re.match(r'\d{4}-\d{2}-\d{2}', sample_str):
                    return DataType.DATE
        
        # Check for ID columns by name (primary keys, foreign keys, etc.)
        if any(hint in column_lower for hint in ['id', 'key', 'pk', 'fk', '_id']):
            # Check if numeric ID
            if pd.api.types.is_numeric_dtype(series):
                # If all values are whole numbers, treat as ID
                if pd.api.types.is_integer_dtype(series) or series.dropna().apply(lambda x: float(x).is_integer()).all():
                    return DataType.ID
            # Check if string-based ID pattern
            string_series = series.astype(str)
            sample_values = string_series.head(20).tolist()
            id_matches = sum(1 for val in sample_values if self.id_pattern.match(val))
            if id_matches / len(sample_values) > 0.7:
                return DataType.ID
            
        # Check for numeric types (but not IDs)
        if pd.api.types.is_numeric_dtype(series):
            return DataType.NUMERIC
            
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATE
            
        # Check for boolean
        if pd.api.types.is_bool_dtype(series) or series.isin([True, False, 'true', 'false', 'True', 'False', 1, 0]).all():
            return DataType.BOOLEAN
            
        # For string data, use pattern matching
        string_series = series.astype(str)
        sample_values = string_series.head(20).tolist()
        
        # Email detection
        email_matches = sum(1 for val in sample_values if self.email_pattern.match(val))
        if email_matches / len(sample_values) > 0.7:
            return DataType.EMAIL
            
        # Phone detection
        phone_matches = sum(1 for val in sample_values if self.phone_pattern.match(val))
        if phone_matches / len(sample_values) > 0.7:
            return DataType.PHONE
            
        # ID detection for string patterns
        id_matches = sum(1 for val in sample_values if self.id_pattern.match(val))
        if id_matches / len(sample_values) > 0.7:
            return DataType.ID
            
        # Other column name hints
        if any(hint in column_lower for hint in ['name', 'first', 'last', 'full_name']):
            return DataType.NAME
        elif any(hint in column_lower for hint in ['address', 'street', 'city', 'location']):
            return DataType.ADDRESS
        elif any(hint in column_lower for hint in ['price', 'cost', 'amount', 'salary', 'revenue']):
            return DataType.CURRENCY
        elif any(hint in column_lower for hint in ['date', 'time', 'created', 'updated', 'timestamp', 'modified', 'last_update']):
            return DataType.DATE
            
        # Check if categorical (low unique count relative to total)
        if series.nunique() / len(series) < 0.1 and series.nunique() < 50:
            return DataType.CATEGORICAL
            
        return DataType.TEXT
    
    def _add_type_specific_stats(self, profile: ColumnProfile, series: pd.Series):
        """Add statistics specific to the detected data type"""
        
        if profile.data_type == DataType.NUMERIC:
            profile.min_value = float(series.min())
            profile.max_value = float(series.max())
            profile.mean_value = float(series.mean())
            profile.std_value = float(series.std())
            
            # Infer decimal places from the data
            if hasattr(self, 'generator'):
                profile.decimal_places = self.generator._infer_decimal_places(series, profile)
            else:
                profile.decimal_places = 2  # Default fallback
            
        elif profile.data_type == DataType.ID:
            # For ID columns, store format information
            if pd.api.types.is_numeric_dtype(series):
                # Numeric IDs
                profile.min_value = int(series.min())
                profile.max_value = int(series.max())
                profile.pattern = "numeric"
            else:
                # String IDs - analyze pattern
                string_series = series.astype(str)
                sample_values = string_series.head(10).tolist()
                
                # Detect common patterns
                if all(val.isdigit() for val in sample_values if val):
                    profile.pattern = "numeric_string"
                    profile.min_value = min(len(val) for val in sample_values if val)
                    profile.max_value = max(len(val) for val in sample_values if val)
                elif all(val.startswith(tuple(['ID', 'USR', 'CUST', 'ORD'])) for val in sample_values if val):
                    profile.pattern = "prefixed"
                    # Extract common prefix
                    prefixes = set(val[:3] for val in sample_values if len(val) >= 3)
                    if len(prefixes) == 1:
                        profile.pattern = f"prefix_{list(prefixes)[0]}"
                else:
                    profile.pattern = "mixed"
            
        elif profile.data_type == DataType.CATEGORICAL:
            profile.categories = series.value_counts().head(20).index.tolist()
            
        elif profile.data_type == DataType.DATE:
            try:
                date_series = pd.to_datetime(series, errors='coerce').dropna()
                if len(date_series) > 0:
                    profile.date_range = (date_series.min(), date_series.max())
            except:
                pass

class SyntheticDataGenerator:
    """Main class for generating synthetic data using Ollama"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.ollama_client = ollama.Client()
        self.inferrer = DataTypeInferrer(self.ollama_client)
        
        if config.seed:
            np.random.seed(config.seed)
        
        # Set up decimal precision for financial calculations
        getcontext().prec = 10
        getcontext().rounding = ROUND_HALF_UP
        
        # Define column precision rules
        self.default_precision_rules = {
            # Financial patterns
            'price': 2, 'cost': 2, 'amount': 2, 'fee': 2, 'charge': 2,
            'salary': 2, 'wage': 2, 'revenue': 2, 'profit': 2, 'loss': 2,
            'balance': 2, 'payment': 2, 'refund': 2, 'discount': 2,
            'tax': 2, 'vat': 2, 'commission': 2, 'bonus': 2,
            
            # Measurement patterns
            'weight': 3, 'height': 2, 'length': 2, 'width': 2, 'depth': 2,
            'volume': 3, 'area': 2, 'distance': 2, 'speed': 1, 'temperature': 1,
            
            # Percentage patterns
            'rate': 3, 'percent': 2, 'ratio': 3, 'score': 1, 'rating': 1,
            
            # Count patterns (should be integers)
            'count': 0, 'quantity': 0, 'num': 0, 'number': 0, 'total': 0,
            'id': 0, 'key': 0, 'index': 0, 'rank': 0, 'position': 0,
            'age': 0, 'year': 0, 'month': 0, 'day': 0, 'hour': 0, 'minute': 0
        }
        
        # Pass reference to self for decimal inference
        self.inferrer.generator = self
    
    def _infer_decimal_places(self, series: pd.Series, profile: ColumnProfile) -> int:
        """Infer the number of decimal places from numeric data"""
        
        # Convert to float to handle any numeric type
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(numeric_series) == 0:
            return self.config.max_decimal_places
        
        decimal_places_list = []
        
        for value in numeric_series.head(20):  # Sample first 20 values
            if pd.isna(value):
                continue
                
            # Convert to string to count decimal places
            value_str = str(float(value))
            
            # Handle scientific notation
            if 'e' in value_str.lower():
                # For scientific notation, use default
                decimal_places_list.append(self.config.max_decimal_places)
                continue
            
            if '.' in value_str:
                # Count digits after decimal point, excluding trailing zeros
                decimal_part = value_str.split('.')[1].rstrip('0')
                decimal_places_list.append(len(decimal_part))
            else:
                # Integer value
                decimal_places_list.append(0)
        
        if not decimal_places_list:
            return self.config.max_decimal_places
        
        # Use the most common number of decimal places
        from collections import Counter
        most_common = Counter(decimal_places_list).most_common(1)[0][0]
        
        # Apply column-specific precision rules
        inferred_precision = min(most_common, self.config.max_decimal_places)
        return self._apply_precision_rules(profile.name, inferred_precision)
    
    def _apply_precision_rules(self, column_name: str, inferred_precision: int) -> int:
        """Apply column-specific precision rules"""
        
        column_lower = column_name.lower()
        
        # Check custom rules first
        if (self.config.column_precision_rules and 
            column_name in self.config.column_precision_rules):
            return self.config.column_precision_rules[column_name]
        
        # Check default precision rules
        for pattern, precision in self.default_precision_rules.items():
            if pattern in column_lower:
                return precision
        
        # Return inferred precision if no rules match
        return inferred_precision
    
    def analyze_csv(self, csv_path: str) -> List[ColumnProfile]:
        """Analyze a CSV file and create column profiles"""
        
        console.print(f"[blue]Analyzing CSV file: {csv_path}[/blue]")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        console.print(f"Found {len(df)} rows and {len(df.columns)} columns")
        
        # Analyze each column
        profiles = []
        with Progress() as progress:
            task = progress.add_task("[green]Analyzing columns...", total=len(df.columns))
            
            for column in df.columns:
                profile = self.inferrer.infer_column_type(df[column], column)
                profiles.append(profile)
                progress.update(task, advance=1)
                
        return profiles
    
    def generate_synthetic_data(self, profiles: List[ColumnProfile]) -> pd.DataFrame:
        """Generate synthetic data based on column profiles"""
        
        console.print(f"[blue]Generating {self.config.num_rows} rows of synthetic data[/blue]")
        
        synthetic_data = {}
        
        with Progress() as progress:
            task = progress.add_task("[green]Generating data...", total=len(profiles))
            
            for profile in profiles:
                column_data = self._generate_column_data(profile)
                synthetic_data[profile.name] = column_data
                progress.update(task, advance=1)
        
        return pd.DataFrame(synthetic_data)
    
    def _generate_column_data(self, profile: ColumnProfile) -> List[Any]:
        """Generate synthetic data for a single column"""
        
        # Debug logging for date columns
        if 'update' in profile.name.lower() or 'date' in profile.name.lower() or 'time' in profile.name.lower():
            logger.info(f"Generating {profile.name} as {profile.data_type}")
        
        if profile.data_type == DataType.NUMERIC:
            return self._generate_numeric_data(profile)
        elif profile.data_type == DataType.CATEGORICAL:
            return self._generate_categorical_data(profile)
        elif profile.data_type == DataType.DATE:
            return self._generate_date_data(profile)
        elif profile.data_type == DataType.EMAIL:
            return self._generate_email_data(profile)
        elif profile.data_type == DataType.PHONE:
            return self._generate_phone_data(profile)
        elif profile.data_type == DataType.NAME:
            return self._generate_name_data(profile)
        elif profile.data_type == DataType.ADDRESS:
            return self._generate_address_data(profile)
        elif profile.data_type == DataType.ID:
            return self._generate_id_data(profile)
        elif profile.data_type == DataType.BOOLEAN:
            return self._generate_boolean_data(profile)
        elif profile.data_type == DataType.CURRENCY:
            return self._generate_currency_data(profile)
        else:
            return self._generate_text_data(profile)
    
    def _generate_numeric_data(self, profile: ColumnProfile) -> List[Union[int, float]]:
        """Generate numeric data using statistical properties"""
        
        # Check if original data was integers
        is_integer_data = False
        if hasattr(profile, 'sample_values') and profile.sample_values:
            # Check if all sample values are integers
            try:
                is_integer_data = all(
                    float(val).is_integer() 
                    for val in profile.sample_values 
                    if val is not None and not pd.isna(val)
                )
            except (ValueError, TypeError):
                is_integer_data = False
        
        if profile.mean_value and profile.std_value:
            # Use normal distribution based on original statistics
            data = np.random.normal(profile.mean_value, profile.std_value, self.config.num_rows)
            
            # Clip to original range if available
            if profile.min_value is not None and profile.max_value is not None:
                data = np.clip(data, profile.min_value, profile.max_value)
        else:
            # Fallback to uniform distribution
            min_val = profile.min_value or 0
            max_val = profile.max_value or 100
            data = np.random.uniform(min_val, max_val, self.config.num_rows)
        
        # Apply proper formatting based on original data
        if is_integer_data:
            data = np.round(data).astype(int)
            return data.tolist()
        else:
            # Apply decimal place formatting
            if hasattr(profile, 'decimal_places') and profile.decimal_places is not None:
                if self.config.preserve_decimal_places:
                    # Use the inferred decimal places
                    decimal_places = profile.decimal_places
                else:
                    # Use the configured maximum
                    decimal_places = self.config.max_decimal_places
                
                # Apply fixed-point arithmetic for financial data if enabled
                if self.config.use_fixed_point and self._is_financial_column(profile.name):
                    return self._apply_fixed_point_precision(data, decimal_places)
                else:
                    # Round to the specified number of decimal places
                    data = np.round(data, decimal_places)
            else:
                # Default to 2 decimal places
                decimal_places = self.config.max_decimal_places
                data = np.round(data, decimal_places)
            
            return data.tolist()
    
    def _is_financial_column(self, column_name: str) -> bool:
        """Check if column contains financial data"""
        financial_patterns = [
            'price', 'cost', 'amount', 'fee', 'charge', 'salary', 'wage',
            'revenue', 'profit', 'loss', 'balance', 'payment', 'refund',
            'discount', 'tax', 'vat', 'commission', 'bonus', 'currency'
        ]
        column_lower = column_name.lower()
        return any(pattern in column_lower for pattern in financial_patterns)
    
    def _apply_fixed_point_precision(self, data: np.ndarray, decimal_places: int) -> List[float]:
        """Apply fixed-point decimal precision for financial accuracy"""
        result = []
        
        for value in data:
            # Convert to Decimal for precise arithmetic
            decimal_value = Decimal(str(value)).quantize(
                Decimal('0.' + '0' * decimal_places),
                rounding=ROUND_HALF_UP
            )
            # Convert back to float for compatibility
            result.append(float(decimal_value))
        
        return result
    
    def _generate_categorical_data(self, profile: ColumnProfile) -> List[str]:
        """Generate categorical data based on original distribution"""
        
        if profile.categories:
            # Clean and split original categories that might be comma-separated
            clean_categories = []
            for category in profile.categories:
                if isinstance(category, str) and ',' in category:
                    # Split comma-separated values and take individual items
                    items = [item.strip() for item in category.split(',')]
                    clean_categories.extend(items)
                else:
                    clean_categories.append(str(category).strip())
            
            # Remove duplicates and empty values
            clean_categories = list(set([cat for cat in clean_categories if cat]))
            
            if len(clean_categories) >= 3:
                # Use 80% original categories, 20% variations
                result = []
                for _ in range(self.config.num_rows):
                    if np.random.random() < 0.8:
                        # Use original category (single item only)
                        result.append(np.random.choice(clean_categories))
                    else:
                        # Generate slight variation
                        base_category = np.random.choice(clean_categories)
                        variation = self._create_category_variation(base_category)
                        result.append(variation)
                return result
            else:
                # Too few categories, generate more using LLM
                return self._generate_with_llm(profile, "single categorical values similar to these categories")
        else:
            # Generate using LLM if no categories available
            return self._generate_with_llm(profile, "single categorical values")
    
    def _generate_date_data(self, profile: ColumnProfile) -> List[str]:
        """Generate date data within the original range"""
        
        # Analyze the original date format from sample values
        date_format = self._infer_date_format(profile.sample_values)
        
        if profile.date_range:
            start_date, end_date = profile.date_range
            date_range_days = (end_date - start_date).days
            
            dates = []
            for _ in range(self.config.num_rows):
                # Generate random date within range
                random_days = np.random.randint(0, max(1, date_range_days))
                random_date = start_date + timedelta(days=random_days)
                
                # Add random time component if original data had timestamps
                if 'timestamp' in date_format or 'time' in date_format:
                    # Add random hours, minutes, seconds
                    random_hours = np.random.randint(0, 24)
                    random_minutes = np.random.randint(0, 60) 
                    random_seconds = np.random.randint(0, 60)
                    random_date = random_date.replace(
                        hour=random_hours, 
                        minute=random_minutes, 
                        second=random_seconds
                    )
                
                # Format according to detected pattern
                formatted_date = self._format_date_value(random_date, date_format)
                dates.append(formatted_date)
            
            return dates
        else:
            # Generate recent dates when no range is available
            base_date = datetime.now() - timedelta(days=365)
            dates = []
            
            for _ in range(self.config.num_rows):
                random_days = np.random.randint(0, 365)
                random_hours = np.random.randint(0, 24)
                random_minutes = np.random.randint(0, 60)
                random_seconds = np.random.randint(0, 60)
                
                random_date = base_date + timedelta(
                    days=random_days,
                    hours=random_hours,
                    minutes=random_minutes,
                    seconds=random_seconds
                )
                
                # Format according to detected pattern
                formatted_date = self._format_date_value(random_date, date_format)
                dates.append(formatted_date)
                
            return dates
    
    def _infer_date_format(self, sample_values: List[Any]) -> str:
        """Infer the date format from sample values"""
        
        if not sample_values:
            return 'datetime'
        
        # Analyze first few sample values to determine format
        for sample in sample_values[:3]:
            if sample is None or pd.isna(sample):
                continue
                
            sample_str = str(sample).strip()
            
            # Check for common timestamp patterns
            if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', sample_str):
                return 'datetime'  # YYYY-MM-DD HH:MM:SS
            elif re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', sample_str):
                return 'iso_datetime'  # ISO format
            elif re.match(r'\d{2}:\d{2}:\d{2}', sample_str):
                return 'time_only'  # HH:MM:SS
            elif re.match(r'\d{4}-\d{2}-\d{2}', sample_str):
                return 'date_only'  # YYYY-MM-DD
            elif re.match(r'\d{2}/\d{2}/\d{4}', sample_str):
                return 'date_us'  # MM/DD/YYYY
            elif re.match(r'\d{2}-\d{2}-\d{4}', sample_str):
                return 'date_dashed'  # MM-DD-YYYY
        
        # Default to datetime if can't determine
        return 'datetime'
    
    def _format_date_value(self, date_obj: datetime, date_format: str) -> str:
        """Format date according to detected pattern"""
        
        format_patterns = {
            'datetime': '%Y-%m-%d %H:%M:%S',
            'iso_datetime': '%Y-%m-%dT%H:%M:%S', 
            'time_only': '%H:%M:%S',
            'date_only': '%Y-%m-%d',
            'date_us': '%m/%d/%Y',
            'date_dashed': '%m-%d-%Y'
        }
        
        pattern = format_patterns.get(date_format, '%Y-%m-%d %H:%M:%S')
        return date_obj.strftime(pattern)
    
    def _generate_email_data(self, profile: ColumnProfile) -> List[str]:
        """Generate realistic email addresses"""
        
        domains = ['example.com', 'test.org', 'sample.net', 'demo.co', 'dev.io']
        emails = []
        
        for _ in range(self.config.num_rows):
            username = f"user{np.random.randint(1000, 9999)}"
            domain = np.random.choice(domains)
            emails.append(f"{username}@{domain}")
            
        return emails
    
    def _generate_phone_data(self, profile: ColumnProfile) -> List[str]:
        """Generate phone numbers"""
        
        phones = []
        for _ in range(self.config.num_rows):
            phone = f"+1{np.random.randint(100, 999)}{np.random.randint(100, 999)}{np.random.randint(1000, 9999)}"
            phones.append(phone)
            
        return phones
    
    def _generate_name_data(self, profile: ColumnProfile) -> List[str]:
        """Generate names using LLM"""
        return self._generate_with_llm(profile, "realistic person names")
    
    def _generate_address_data(self, profile: ColumnProfile) -> List[str]:
        """Generate addresses using LLM"""
        return self._generate_with_llm(profile, "realistic addresses")
    
    def _generate_id_data(self, profile: ColumnProfile) -> List[Union[int, str]]:
        """Generate ID values based on the original pattern"""
        
        ids = []
        
        if profile.pattern == "numeric":
            # Generate integer IDs
            if profile.min_value is not None and profile.max_value is not None:
                # Use range from original data
                min_id = int(profile.min_value)
                max_id = int(profile.max_value)
                # Extend range for synthetic data to avoid conflicts
                range_size = max_id - min_id
                start_id = max_id + 1
                end_id = start_id + self.config.num_rows
                ids = list(range(start_id, end_id))
            else:
                # Default range for integer IDs
                start_id = 100000
                ids = list(range(start_id, start_id + self.config.num_rows))
            
        elif profile.pattern == "numeric_string":
            # Generate string IDs that look like numbers
            id_length = profile.max_value or 8
            for i in range(self.config.num_rows):
                # Generate a random number with fixed length
                id_num = np.random.randint(10**(id_length-1), 10**id_length)
                ids.append(str(id_num))
                
        elif profile.pattern and profile.pattern.startswith("prefix_"):
            # Extract prefix and generate with that prefix
            prefix = profile.pattern.split("_", 1)[1]
            for i in range(self.config.num_rows):
                id_val = f"{prefix}{np.random.randint(100000, 999999)}"
                ids.append(id_val)
                
        else:
            # Fallback: analyze sample values to determine pattern
            if profile.sample_values:
                sample = str(profile.sample_values[0]) if profile.sample_values[0] is not None else "ID000001"
                
                # Try to detect pattern from sample
                if sample.isdigit():
                    # Pure numeric string
                    length = len(sample)
                    for i in range(self.config.num_rows):
                        id_num = np.random.randint(10**(length-1), 10**length)
                        ids.append(str(id_num))
                        
                elif any(char.isdigit() for char in sample):
                    # Mixed pattern - try to preserve prefix/suffix
                    import re
                    # Extract prefix (letters at start)
                    prefix_match = re.match(r'^([A-Za-z]+)', sample)
                    prefix = prefix_match.group(1) if prefix_match else "ID"
                    
                    for i in range(self.config.num_rows):
                        id_val = f"{prefix}{np.random.randint(100000, 999999)}"
                        ids.append(id_val)
                else:
                    # Pure string - use structured ID generation instead of LLM for consistency
                    for i in range(self.config.num_rows):
                        id_val = f"ID{str(i + 100000).zfill(6)}"
                        ids.append(id_val)
            else:
                # No sample data - use default pattern
                for i in range(self.config.num_rows):
                    id_val = f"ID{np.random.randint(100000, 999999)}"
                    ids.append(id_val)
        
        return ids
    
    def _generate_boolean_data(self, profile: ColumnProfile) -> List[bool]:
        """Generate boolean data"""
        return np.random.choice([True, False], self.config.num_rows).tolist()
    
    def _generate_currency_data(self, profile: ColumnProfile) -> List[float]:
        """Generate currency values"""
        
        # Generate realistic currency values
        if profile.min_value and profile.max_value:
            min_val = max(0, profile.min_value)
            max_val = profile.max_value
        else:
            min_val = 10
            max_val = 10000
            
        # Use log-normal distribution for realistic currency distribution
        mu = np.log((min_val + max_val) / 2)
        sigma = 0.5
        
        values = np.random.lognormal(mu, sigma, self.config.num_rows)
        values = np.clip(values, min_val, max_val)
        
        # Determine decimal places for currency
        if hasattr(profile, 'decimal_places') and profile.decimal_places is not None:
            decimal_places = profile.decimal_places
        else:
            # Currency typically uses 2 decimal places
            decimal_places = 2
            
        return [round(val, decimal_places) for val in values]
    
    def _generate_text_data(self, profile: ColumnProfile) -> List[str]:
        """Generate text data using LLM"""
        return self._generate_with_llm(profile, "realistic text content")
    
    def _generate_with_llm(self, profile: ColumnProfile, data_description: str) -> List[str]:
        """Generate data using Ollama LLM"""
        
        try:
            # Create a prompt based on the sample values
            sample_str = ", ".join(str(val) for val in profile.sample_values[:5])
            
            # Create data type specific prompt
            if profile.data_type == DataType.ID:
                prompt = f"""Generate {min(50, self.config.num_rows)} unique identifier values following this pattern: {sample_str}

CRITICAL RULES:
- Generate ONLY the ID values, nothing else
- NO markdown formatting (no **, `, etc.)
- NO explanations or labels
- NO bullet points or numbering
- Each ID on a separate line
- Follow the exact same format as the examples
- For numeric IDs: use only integers
- For string IDs: maintain the same prefix/pattern

Examples: {sample_str}

Generate similar IDs:"""
            
            elif profile.data_type == DataType.NAME:
                prompt = f"""Generate exactly {min(50, self.config.num_rows)} individual person names, one per line.

FORMAT: Follow this exact pattern from examples: {sample_str}

STRICT RULES:
- ONE name per line only
- NO lists, NO commas between names
- NO numbering (1., 2., etc.)
- NO bullet points (•, -, *)
- NO explanations or extra text
- Each line = exactly one complete name
- If examples show "John Smith", generate names like "David Brown" (not "David, Brown, John, Smith")
- Use the column/field name for context

Examples of the format to follow:
{sample_str}

Generate individual names in the same format:"""
            
            elif profile.data_type == DataType.ADDRESS:
                prompt = f"""Generate exactly {min(50, self.config.num_rows)} individual addresses, one per line.

FORMAT: Follow this exact pattern from examples: {sample_str}

STRICT RULES:
- ONE address per line only
- NO lists, NO commas between addresses  
- NO numbering (1., 2., etc.)
- NO bullet points (•, -, *)
- NO explanations or extra text
- Each line = exactly one complete address
- Use realistic street names and numbers

Examples of the format to follow:
{sample_str}

Generate individual addresses in the same format:"""
            
            else:
                # Generic prompt for other data types
                prompt = f"""Generate {min(50, self.config.num_rows)} realistic {data_description} similar to: {sample_str}

CRITICAL RULES:
- Generate ONLY single values, nothing else
- NO comma-separated lists or multiple items per line
- NO markdown formatting (no **, `, etc.)
- NO explanations or labels
- NO bullet points or numbering
- NO headers or titles
- Each value on a separate line
- Each line should contain ONLY ONE item
- Follow the exact same format/pattern as examples
- Make data realistic but varied
- If examples contain lists, pick only ONE item per generated value

Examples: {sample_str}

Generate similar single values:"""
            
            response = self.ollama_client.generate(
                model=self.config.ollama_model,
                prompt=prompt,
                stream=False
            )
            
            # Clean the response to remove any markdown or formatting
            raw_response = response['response'].strip()
            
            # Split into lines and clean each value
            generated_values = []
            for line in raw_response.split('\n'):
                cleaned_line = self._clean_llm_output(line.strip())
                if cleaned_line and len(cleaned_line) > 0:
                    # Check if this line contains multiple items (comma-separated)
                    if ',' in cleaned_line and profile.data_type in [DataType.NAME, DataType.CATEGORICAL, DataType.TEXT]:
                        # Split and take each item separately
                        items = [item.strip() for item in cleaned_line.split(',')]
                        for item in items:
                            if item:
                                processed_value = self._process_generated_value(item, profile.data_type)
                                if processed_value:
                                    generated_values.append(processed_value)
                    else:
                        # Single item, process normally
                        processed_value = self._process_generated_value(cleaned_line, profile.data_type)
                        if processed_value:
                            generated_values.append(processed_value)
            
            # Remove duplicates while preserving order
            generated_values = list(dict.fromkeys(generated_values))
            
            # Validate that we have good data
            if not generated_values:
                raise ValueError("No valid data generated from LLM")
            
            # If we don't have enough unique values, add some variations
            if len(generated_values) < min(20, self.config.num_rows):
                generated_values = self._expand_generated_values(generated_values, profile)
            
            # Extend to required length by cycling through generated values
            if len(generated_values) < self.config.num_rows:
                cycle_count = self.config.num_rows // len(generated_values) + 1
                generated_values = (generated_values * cycle_count)[:self.config.num_rows]
            else:
                generated_values = generated_values[:self.config.num_rows]
                
            return generated_values
            
        except Exception as e:
            logger.warning(f"LLM generation failed for {profile.name}: {e}")
            # Fallback to simple generation
            return [f"{profile.name}_{i}" for i in range(self.config.num_rows)]
    
    def _clean_llm_output(self, text: str) -> str:
        """Clean LLM output to remove markdown and formatting"""
        
        if not text:
            return ""
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic*
        text = re.sub(r'`(.*?)`', r'\1', text)        # Remove `code`
        text = re.sub(r'#{1,6}\s*', '', text)         # Remove headers
        text = re.sub(r'^\s*[-*+]\s*', '', text)      # Remove bullet points
        text = re.sub(r'^\s*\d+\.\s*', '', text)      # Remove numbered lists
        text = re.sub(r'^\s*\|\s*', '', text)         # Remove table formatting
        text = re.sub(r'\s*\|\s*$', '', text)         # Remove trailing table formatting
        
        # Remove common prefixes that LLMs add
        prefixes_to_remove = [
            "Here are", "Here's", "Generated", "Example", "Sample",
            "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",
            "•", "-", "*", "+"
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove quotes if the entire string is quoted
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        
        # Remove any trailing colons or explanatory text
        if ':' in text and any(word in text.lower() for word in ['example', 'sample', 'generated']):
            # Likely contains explanatory text, try to extract just the value
            parts = text.split(':')
            if len(parts) > 1:
                # Take the part after the colon, or the longest part
                text = max(parts, key=len).strip()
        
        return text.strip()
    
    def _process_generated_value(self, value: str, data_type: DataType) -> Optional[str]:
        """Process and validate generated values with length limits"""
        
        if not value or len(value.strip()) == 0:
            return None
        
        value = value.strip()
        
        # Apply length limits based on data type
        max_lengths = {
            DataType.NAME: 100,
            DataType.EMAIL: 255,
            DataType.PHONE: 20,
            DataType.ADDRESS: 200,
            DataType.CATEGORICAL: 100,
            DataType.TEXT: 500,
            DataType.ID: 50
        }
        
        max_length = max_lengths.get(data_type, 255)
        
        # Universal comma removal for all types (LLM should never generate lists)
        if ',' in value:
            value = value.split(',')[0].strip()
        
        # Remove list-like formatting for all types
        value = re.sub(r'^\d+[\.\)]\s*', '', value)  # Remove "1. " or "1) "
        value = re.sub(r'^[-•*]\s*', '', value)  # Remove bullet points
        value = value.strip('"\'')  # Remove quotes
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length].strip()
            # If truncated in the middle of a word, remove the partial word
            if ' ' in value:
                value = ' '.join(value.split()[:-1])
        
        # Type-specific cleaning
        if data_type == DataType.NAME:
            # Names should not have conjunctions or extra words
            value = re.sub(r'\b(and|or|&)\b', '', value, flags=re.IGNORECASE)
            value = ' '.join(value.split())  # Normalize whitespace
            
        elif data_type == DataType.CATEGORICAL:
            # Additional cleaning for categorical values
            value = value.replace('and ', '').replace(' and', '')  # Remove conjunctions
            value = ' '.join(value.split())  # Normalize whitespace
            
        elif data_type == DataType.DATE:
            # Date/timestamp should not be processed by LLM cleaning
            # Return the value as-is if it looks like a date, otherwise None
            if re.match(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|\d{2}/\d{2}/\d{4}', value):
                return value
            else:
                return None
        
        # Final length check
        if len(value) < 1 or len(value) > max_length:
            return None
            
        return value
    
    def _expand_generated_values(self, values: List[str], profile: ColumnProfile) -> List[str]:
        """Expand the list of generated values with variations"""
        
        if not values:
            return values
        
        expanded = values.copy()
        
        # Add simple variations based on data type
        if profile.data_type == DataType.CATEGORICAL:
            # For categorical data, create slight variations
            for value in values[:5]:  # Only vary first 5 to avoid too many
                variation = self._create_category_variation(value)
                if variation and variation not in expanded:
                    expanded.append(variation)
        
        elif profile.data_type == DataType.NAME:
            # For names, create variations by changing parts
            for value in values[:3]:
                if ' ' in value:
                    parts = value.split()
                    if len(parts) == 2:
                        # Create initials version
                        initial_version = f"{parts[0][0]}. {parts[1]}"
                        if initial_version not in expanded:
                            expanded.append(initial_version)
        
        return expanded
    
    def _create_category_variation(self, category: str) -> str:
        """Create a slight variation of a category"""
        
        # Clean the input category first
        category = category.strip()
        if ',' in category:
            category = category.split(',')[0].strip()
        
        variations = {
            'Deleted Scenes': ['Bonus Scenes', 'Cut Scenes', 'Extended Scenes', 'Outtakes'],
            'Behind the Scenes': ['Making Of', 'Production Notes', 'Behind-the-Camera', 'Documentary'],
            'Commentaries': ['Audio Commentary', 'Director Commentary', 'Cast Commentary', 'Commentary Track'],
            'Trailers': ['Previews', 'Teasers', 'Promotional Videos', 'Movie Trailer'],
            'Triples': ['Triple Features', 'Multi-Part', 'Three-Pack', 'Collection']
        }
        
        # Exact match
        if category in variations:
            return np.random.choice(variations[category])
        
        # Partial match (case insensitive)
        for key, values in variations.items():
            if key.lower() in category.lower() or category.lower() in key.lower():
                return np.random.choice(values)
        
        # Generic variations for unknown categories
        if len(category.split()) > 1:
            # Multi-word: try rearranging or shortening
            words = category.split()
            if len(words) == 2:
                return f"{words[1]} {words[0]}"  # Swap words
            elif len(words) > 2:
                return ' '.join(words[:2])  # Take first two words
        
        # Single word or fallback: add descriptive suffix
        suffixes = ['Extended', 'Special', 'Premium', 'Classic', 'Deluxe', 'Enhanced']
        return f"{category} {np.random.choice(suffixes)}"
    
    def save_data(self, df: pd.DataFrame, output_path: str):
        """Save generated data to file"""
        
        output_path = Path(output_path)
        
        if self.config.output_format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif self.config.output_format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif self.config.output_format.lower() == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")
            
        console.print(f"[green]Synthetic data saved to: {output_path}[/green]")
    
    def generate_report(self, original_profiles: List[ColumnProfile], synthetic_df: pd.DataFrame) -> str:
        """Generate a comparison report between original and synthetic data"""
        
        report = []
        report.append("# Synthetic Data Generation Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"Rows generated: {len(synthetic_df)}\n")
        report.append(f"Configuration: {asdict(self.config)}\n\n")
        
        # Column comparison
        report.append("## Column Analysis\n")
        
        table = Table(title="Column Comparison")
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Original Unique", style="yellow")
        table.add_column("Synthetic Unique", style="green")
        table.add_column("Quality Score", style="blue")
        
        for profile in original_profiles:
            if profile.name in synthetic_df.columns:
                synthetic_unique = synthetic_df[profile.name].nunique()
                
                # Calculate quality score
                unique_ratio = min(synthetic_unique / profile.unique_count, 1.0) if profile.unique_count > 0 else 1.0
                quality_score = f"{unique_ratio:.2%}"
                
                table.add_row(
                    profile.name,
                    profile.data_type.value,
                    str(profile.unique_count),
                    str(synthetic_unique),
                    quality_score
                )
        
        console.print(table)
        
        return "\n".join(report)

def main():
    """Main CLI interface"""
    app = typer.Typer()
    
    @app.command()
    def generate(
        csv_path: str = typer.Argument(..., help="Path to the input CSV file"),
        output_path: str = typer.Option("synthetic_data.csv", help="Output file path"),
        num_rows: int = typer.Option(1000, help="Number of rows to generate"),
        model: str = typer.Option("granite4:latest", help="Ollama model to use"),
        format: str = typer.Option("csv", help="Output format (csv, json, parquet)"),
        seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
        config_file: Optional[str] = typer.Option(None, help="YAML configuration file"),
        max_decimal_places: int = typer.Option(2, help="Maximum decimal places for numeric values"),
        preserve_decimals: bool = typer.Option(True, help="Preserve original decimal precision"),
    ):
        """Generate synthetic data from a CSV file using Ollama"""
        
        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            config = GenerationConfig(**config_dict)
        else:
            config = GenerationConfig(
                num_rows=num_rows,
                ollama_model=model,
                output_format=format,
                seed=seed,
                max_decimal_places=max_decimal_places,
                preserve_decimal_places=preserve_decimals
            )
        
        # Create generator
        generator = SyntheticDataGenerator(config)
        
        # Analyze input CSV
        profiles = generator.analyze_csv(csv_path)
        
        # Generate synthetic data
        synthetic_df = generator.generate_synthetic_data(profiles)
        
        # Save results
        generator.save_data(synthetic_df, output_path)
        
        # Generate report
        report = generator.generate_report(profiles, synthetic_df)
        
        console.print("[green]✅ Synthetic data generation completed successfully![/green]")
    
    @app.command()
    def analyze(
        csv_path: str = typer.Argument(..., help="Path to the CSV file to analyze"),
    ):
        """Analyze a CSV file and show data type inference results"""
        
        config = GenerationConfig(num_rows=0)  # We're just analyzing
        generator = SyntheticDataGenerator(config)
        
        profiles = generator.analyze_csv(csv_path)
        
        # Display analysis results
        table = Table(title="CSV Analysis Results")
        table.add_column("Column", style="cyan")
        table.add_column("Detected Type", style="magenta")
        table.add_column("Unique Values", style="yellow")
        table.add_column("Null %", style="red")
        table.add_column("Sample Values", style="green")
        
        for profile in profiles:
            sample_str = ", ".join(str(val)[:20] for val in profile.sample_values[:3])
            table.add_row(
                profile.name,
                profile.data_type.value,
                str(profile.unique_count),
                f"{profile.null_percentage:.1f}%",
                sample_str
            )
        
        console.print(table)
    
    app()

if __name__ == "__main__":
    main()