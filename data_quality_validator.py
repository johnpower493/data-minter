#!/usr/bin/env python3
"""
Data Quality Validator for Synthetic Data

Validates the quality of generated synthetic data by comparing
statistical properties with the original dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import scipy.stats as stats

console = Console()

@dataclass
class QualityMetric:
    """Represents a quality metric comparison"""
    metric_name: str
    original_value: Any
    synthetic_value: Any
    score: float  # 0-1 where 1 is perfect
    status: str   # "pass", "warning", "fail"
    threshold: float = 0.8

class DataQualityValidator:
    """Validates synthetic data quality against original data"""
    
    def __init__(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame):
        self.original_df = original_df
        self.synthetic_df = synthetic_df
        self.metrics = []
        
    def validate_all(self) -> Dict[str, Any]:
        """Run all quality validations"""
        
        console.print("[bold blue]🔍 Running Data Quality Validation[/bold blue]\n")
        
        results = {
            "overall_score": 0.0,
            "column_scores": {},
            "metrics": [],
            "recommendations": []
        }
        
        # Validate each column
        for column in self.original_df.columns:
            if column in self.synthetic_df.columns:
                column_metrics = self._validate_column(column)
                results["column_scores"][column] = column_metrics
                results["metrics"].extend(column_metrics)
        
        # Calculate overall score
        if results["metrics"]:
            results["overall_score"] = np.mean([m.score for m in results["metrics"]])
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results["metrics"])
        
        # Display results
        self._display_results(results)
        
        return results
    
    def _validate_column(self, column: str) -> List[QualityMetric]:
        """Validate a single column"""
        
        original_series = self.original_df[column].dropna()
        synthetic_series = self.synthetic_df[column].dropna()
        
        metrics = []
        
        # Basic metrics
        metrics.extend(self._validate_basic_stats(column, original_series, synthetic_series))
        
        # Type-specific metrics
        if pd.api.types.is_numeric_dtype(original_series):
            metrics.extend(self._validate_numeric_column(column, original_series, synthetic_series))
        elif pd.api.types.is_categorical_dtype(original_series) or original_series.nunique() < 20:
            metrics.extend(self._validate_categorical_column(column, original_series, synthetic_series))
        elif pd.api.types.is_datetime64_any_dtype(original_series):
            metrics.extend(self._validate_datetime_column(column, original_series, synthetic_series))
        else:
            metrics.extend(self._validate_text_column(column, original_series, synthetic_series))
        
        return metrics
    
    def _validate_basic_stats(self, column: str, original: pd.Series, synthetic: pd.Series) -> List[QualityMetric]:
        """Validate basic statistics"""
        
        metrics = []
        
        # Uniqueness ratio
        orig_unique_ratio = original.nunique() / len(original)
        synth_unique_ratio = synthetic.nunique() / len(synthetic)
        uniqueness_score = 1 - abs(orig_unique_ratio - synth_unique_ratio)
        
        metrics.append(QualityMetric(
            metric_name=f"{column}_uniqueness",
            original_value=f"{orig_unique_ratio:.3f}",
            synthetic_value=f"{synth_unique_ratio:.3f}",
            score=max(0, uniqueness_score),
            status="pass" if uniqueness_score >= 0.8 else "warning" if uniqueness_score >= 0.6 else "fail"
        ))
        
        # Length distribution (for string data)
        if original.dtype == 'object':
            orig_avg_length = original.astype(str).str.len().mean()
            synth_avg_length = synthetic.astype(str).str.len().mean()
            
            if orig_avg_length > 0:
                length_score = 1 - abs(orig_avg_length - synth_avg_length) / orig_avg_length
                metrics.append(QualityMetric(
                    metric_name=f"{column}_avg_length",
                    original_value=f"{orig_avg_length:.1f}",
                    synthetic_value=f"{synth_avg_length:.1f}",
                    score=max(0, length_score),
                    status="pass" if length_score >= 0.8 else "warning" if length_score >= 0.6 else "fail"
                ))
        
        return metrics
    
    def _validate_numeric_column(self, column: str, original: pd.Series, synthetic: pd.Series) -> List[QualityMetric]:
        """Validate numeric column statistics"""
        
        metrics = []
        
        # Mean comparison
        orig_mean = original.mean()
        synth_mean = synthetic.mean()
        if orig_mean != 0:
            mean_diff = abs(orig_mean - synth_mean) / abs(orig_mean)
            mean_score = max(0, 1 - mean_diff)
        else:
            mean_score = 1.0 if abs(synth_mean) < 0.001 else 0.0
        
        metrics.append(QualityMetric(
            metric_name=f"{column}_mean",
            original_value=f"{orig_mean:.3f}",
            synthetic_value=f"{synth_mean:.3f}",
            score=mean_score,
            status="pass" if mean_score >= 0.8 else "warning" if mean_score >= 0.6 else "fail"
        ))
        
        # Standard deviation comparison
        orig_std = original.std()
        synth_std = synthetic.std()
        if orig_std != 0:
            std_diff = abs(orig_std - synth_std) / orig_std
            std_score = max(0, 1 - std_diff)
        else:
            std_score = 1.0 if abs(synth_std) < 0.001 else 0.0
        
        metrics.append(QualityMetric(
            metric_name=f"{column}_std",
            original_value=f"{orig_std:.3f}",
            synthetic_value=f"{synth_std:.3f}",
            score=std_score,
            status="pass" if std_score >= 0.8 else "warning" if std_score >= 0.6 else "fail"
        ))
        
        # Distribution comparison using KS test
        try:
            ks_stat, p_value = stats.ks_2samp(original, synthetic)
            # Higher p-value means distributions are more similar
            distribution_score = p_value
            
            metrics.append(QualityMetric(
                metric_name=f"{column}_distribution",
                original_value="baseline",
                synthetic_value=f"p={p_value:.3f}",
                score=distribution_score,
                status="pass" if distribution_score >= 0.05 else "warning" if distribution_score >= 0.01 else "fail"
            ))
        except:
            pass  # Skip if KS test fails
        
        return metrics
    
    def _validate_categorical_column(self, column: str, original: pd.Series, synthetic: pd.Series) -> List[QualityMetric]:
        """Validate categorical column properties"""
        
        metrics = []
        
        # Category coverage
        orig_categories = set(original.unique())
        synth_categories = set(synthetic.unique())
        
        coverage = len(orig_categories.intersection(synth_categories)) / len(orig_categories)
        
        metrics.append(QualityMetric(
            metric_name=f"{column}_category_coverage",
            original_value=f"{len(orig_categories)} categories",
            synthetic_value=f"{coverage:.1%} covered",
            score=coverage,
            status="pass" if coverage >= 0.8 else "warning" if coverage >= 0.6 else "fail"
        ))
        
        # Frequency distribution comparison
        orig_freq = original.value_counts(normalize=True).sort_index()
        synth_freq = synthetic.value_counts(normalize=True).sort_index()
        
        # Calculate overlap in frequency distributions
        common_categories = orig_freq.index.intersection(synth_freq.index)
        if len(common_categories) > 0:
            orig_common_freq = orig_freq[common_categories]
            synth_common_freq = synth_freq[common_categories]
            
            # Calculate similarity using cosine similarity
            dot_product = np.dot(orig_common_freq, synth_common_freq)
            norm_product = np.linalg.norm(orig_common_freq) * np.linalg.norm(synth_common_freq)
            
            if norm_product > 0:
                freq_similarity = dot_product / norm_product
            else:
                freq_similarity = 0.0
            
            metrics.append(QualityMetric(
                metric_name=f"{column}_frequency_similarity",
                original_value="baseline",
                synthetic_value=f"{freq_similarity:.3f}",
                score=freq_similarity,
                status="pass" if freq_similarity >= 0.8 else "warning" if freq_similarity >= 0.6 else "fail"
            ))
        
        return metrics
    
    def _validate_datetime_column(self, column: str, original: pd.Series, synthetic: pd.Series) -> List[QualityMetric]:
        """Validate datetime column properties"""
        
        metrics = []
        
        try:
            orig_dt = pd.to_datetime(original)
            synth_dt = pd.to_datetime(synthetic)
            
            # Date range comparison
            orig_range = (orig_dt.max() - orig_dt.min()).days
            synth_range = (synth_dt.max() - synth_dt.min()).days
            
            if orig_range > 0:
                range_score = 1 - abs(orig_range - synth_range) / orig_range
            else:
                range_score = 1.0 if synth_range == 0 else 0.0
            
            metrics.append(QualityMetric(
                metric_name=f"{column}_date_range",
                original_value=f"{orig_range} days",
                synthetic_value=f"{synth_range} days",
                score=max(0, range_score),
                status="pass" if range_score >= 0.8 else "warning" if range_score >= 0.6 else "fail"
            ))
            
        except:
            pass  # Skip if datetime conversion fails
        
        return metrics
    
    def _validate_text_column(self, column: str, original: pd.Series, synthetic: pd.Series) -> List[QualityMetric]:
        """Validate text column properties"""
        
        metrics = []
        
        # Word count distribution
        orig_word_counts = original.astype(str).str.split().str.len()
        synth_word_counts = synthetic.astype(str).str.split().str.len()
        
        orig_avg_words = orig_word_counts.mean()
        synth_avg_words = synth_word_counts.mean()
        
        if orig_avg_words > 0:
            word_score = 1 - abs(orig_avg_words - synth_avg_words) / orig_avg_words
        else:
            word_score = 1.0 if synth_avg_words == 0 else 0.0
        
        metrics.append(QualityMetric(
            metric_name=f"{column}_avg_word_count",
            original_value=f"{orig_avg_words:.1f}",
            synthetic_value=f"{synth_avg_words:.1f}",
            score=max(0, word_score),
            status="pass" if word_score >= 0.8 else "warning" if word_score >= 0.6 else "fail"
        ))
        
        return metrics
    
    def _generate_recommendations(self, metrics: List[QualityMetric]) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        
        recommendations = []
        
        failed_metrics = [m for m in metrics if m.status == "fail"]
        warning_metrics = [m for m in metrics if m.status == "warning"]
        
        if failed_metrics:
            recommendations.append("🔴 Critical Issues Found:")
            for metric in failed_metrics:
                if "uniqueness" in metric.metric_name:
                    recommendations.append("  • Increase variety in synthetic data generation")
                elif "mean" in metric.metric_name or "std" in metric.metric_name:
                    recommendations.append(f"  • Adjust statistical parameters for {metric.metric_name}")
                elif "distribution" in metric.metric_name:
                    recommendations.append(f"  • Review distribution generation strategy for {metric.metric_name}")
                elif "category_coverage" in metric.metric_name:
                    recommendations.append(f"  • Ensure all original categories are represented in {metric.metric_name}")
        
        if warning_metrics:
            recommendations.append("🟡 Areas for Improvement:")
            for metric in warning_metrics:
                recommendations.append(f"  • Fine-tune generation for {metric.metric_name}")
        
        if not failed_metrics and not warning_metrics:
            recommendations.append("🟢 Excellent data quality! All metrics passed.")
        
        # General recommendations
        recommendations.extend([
            "",
            "💡 General Recommendations:",
            "  • Use larger sample sizes for better statistical accuracy",
            "  • Consider adjusting LLM temperature for more/less variety",
            "  • Validate business rules and constraints separately"
        ])
        
        return recommendations
    
    def _display_results(self, results: Dict[str, Any]):
        """Display validation results in a formatted table"""
        
        # Overall score panel
        score = results["overall_score"]
        score_color = "green" if score >= 0.8 else "yellow" if score >= 0.6 else "red"
        score_emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
        
        console.print(Panel(
            f"{score_emoji} Overall Quality Score: {score:.1%}",
            title="Data Quality Assessment",
            border_style=score_color
        ))
        
        # Detailed metrics table
        table = Table(title="Detailed Quality Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Original", style="blue")
        table.add_column("Synthetic", style="green")
        table.add_column("Score", style="yellow")
        table.add_column("Status", style="bold")
        
        for metric in results["metrics"]:
            status_style = "green" if metric.status == "pass" else "yellow" if metric.status == "warning" else "red"
            status_icon = "✅" if metric.status == "pass" else "⚠️" if metric.status == "warning" else "❌"
            
            table.add_row(
                metric.metric_name,
                str(metric.original_value),
                str(metric.synthetic_value),
                f"{metric.score:.3f}",
                f"[{status_style}]{status_icon} {metric.status}[/{status_style}]"
            )
        
        console.print(table)
        
        # Recommendations
        console.print("\n[bold blue]📋 Recommendations:[/bold blue]")
        for recommendation in results["recommendations"]:
            console.print(recommendation)

def main():
    """CLI interface for data quality validation"""
    import typer
    
    app = typer.Typer()
    
    @app.command()
    def validate(
        original_csv: str = typer.Argument(..., help="Path to original CSV file"),
        synthetic_csv: str = typer.Argument(..., help="Path to synthetic CSV file"),
        output_report: str = typer.Option(None, help="Save report to file"),
    ):
        """Validate synthetic data quality against original data"""
        
        # Load data
        original_df = pd.read_csv(original_csv)
        synthetic_df = pd.read_csv(synthetic_csv)
        
        # Validate
        validator = DataQualityValidator(original_df, synthetic_df)
        results = validator.validate_all()
        
        # Save report if requested
        if output_report:
            import json
            with open(output_report, 'w') as f:
                # Convert QualityMetric objects to dictionaries for JSON serialization
                serializable_results = {
                    "overall_score": results["overall_score"],
                    "column_scores": {},
                    "metrics": [
                        {
                            "metric_name": m.metric_name,
                            "original_value": str(m.original_value),
                            "synthetic_value": str(m.synthetic_value),
                            "score": m.score,
                            "status": m.status
                        } for m in results["metrics"]
                    ],
                    "recommendations": results["recommendations"]
                }
                json.dump(serializable_results, f, indent=2)
            
            console.print(f"\n📄 Report saved to: {output_report}")
    
    app()

if __name__ == "__main__":
    main()