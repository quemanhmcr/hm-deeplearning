"""Data validation utilities for H&M dataset."""

from typing import Dict, List, Any
import pandas as pd
from pathlib import Path

class DataValidator:
    """Enterprise-grade data validation for H&M dataset."""

    def __init__(self, data_root: str = "data/raw"):
        self.data_root = Path(data_root)

        # Define expected schemas
        self.articles_schema = {
            'article_id': 'string',
            'product_code': 'int64',
            'prod_name': 'string',
            'product_type_no': 'int64',
            'product_type_name': 'string',
            'product_group_name': 'string',
            'graphical_appearance_no': 'int64',
            'graphical_appearance_name': 'string',
            'colour_group_code': 'int64',
            'colour_group_name': 'string',
            'perceived_colour_value_id': 'int64',
            'perceived_colour_value_name': 'string',
            'perceived_colour_master_id': 'int64',
            'perceived_colour_master_name': 'string',
            'department_no': 'int64',
            'department_name': 'string',
            'index_code': 'string',
            'index_name': 'string',
            'index_group_no': 'int64',
            'index_group_name': 'string',
            'section_no': 'int64',
            'section_name': 'string',
            'garment_group_no': 'int64',
            'garment_group_name': 'string',
            'detail_desc': 'string'
        }

        self.customers_schema = {
            'customer_id': 'string',
            'FN': 'float64',
            'Active': 'float64',
            'club_member_status': 'string',
            'fashion_news_frequency': 'string',
            'age': 'int64',
            'postal_code': 'string'
        }

        self.transactions_schema = {
            't_dat': 'datetime64[ns]',
            'customer_id': 'string',
            'article_id': 'string',
            'price': 'float64',
            'sales_channel_id': 'int64'
        }

        # Valid values for categorical fields
        self.valid_sales_channels = [1, 2]
        self.valid_club_member_status = ['ACTIVE', 'PRE-CREATION', 'LEFT CLUB']
        self.valid_fashion_news_freq = ['NONE', 'Regularly', 'Monthly', 'None']

    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets with optimized dtypes."""
        datasets = {}

        try:
            # Articles
            datasets['articles'] = pd.read_csv(
                self.data_root / 'articles_filtered.csv',
                dtype=self.articles_schema
            )

            # Customers
            datasets['customers'] = pd.read_csv(
                self.data_root / 'customers_filtered.csv',
                dtype=self.customers_schema
            )

            # Transactions
            datasets['transactions'] = pd.read_csv(
                self.data_root / 'transactions_train_2M.csv',
                dtype={k: v for k, v in self.transactions_schema.items() if v != 'datetime64[ns]'},
                parse_dates=['t_dat']
            )

        except Exception as e:
            raise FileNotFoundError(f"Failed to load datasets: {e}")

        return datasets

    def validate_schema(self, df: pd.DataFrame, schema: Dict[str, str], table_name: str) -> List[str]:
        """Validate DataFrame schema against expected schema."""
        errors = []

        # Check required columns
        missing_cols = set(schema.keys()) - set(df.columns)
        if missing_cols:
            errors.append(f"{table_name}: Missing columns: {missing_cols}")

        # Check for extra columns
        extra_cols = set(df.columns) - set(schema.keys())
        if extra_cols:
            errors.append(f"{table_name}: Unexpected columns: {extra_cols}")

        # Check data types
        for col, expected_type in schema.items():
            if col in df.columns:
                if expected_type == 'datetime64[ns]':
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        try:
                            pd.to_datetime(df[col])
                        except:
                            errors.append(f"{table_name}: {col} is not datetime convertible")
                else:
                    actual_type = str(df[col].dtype)
                    if expected_type not in actual_type:
                        errors.append(f"{table_name}: {col} expected {expected_type}, got {actual_type}")

        return errors

    def validate_business_rules(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Validate business logic and constraints."""
        errors = []

        articles = datasets.get('articles', pd.DataFrame())
        customers = datasets.get('customers', pd.DataFrame())
        transactions = datasets.get('transactions', pd.DataFrame())

        # Validate articles
        if not articles.empty:
            # Check for duplicate article IDs
            if articles['article_id'].duplicated().any():
                dup_count = articles['article_id'].duplicated().sum()
                errors.append(f"articles: Found {dup_count} duplicate article_id values")

            # Validate numeric ranges
            if not articles['product_code'].between(100000, 999999).all():
                errors.append("articles: product_code values outside expected 6-digit range")

        # Validate customers
        if not customers.empty:
            # Check age range
            if not customers['age'].between(16, 99).all():
                invalid_age = customers[~customers['age'].between(16, 99)]['age'].tolist()
                errors.append(f"customers: Invalid age values found: {invalid_age[:5]}...")

            # Validate club member status
            invalid_status = customers[~customers['club_member_status'].isin(self.valid_club_member_status)]
            if not invalid_status.empty:
                unique_invalid = invalid_status['club_member_status'].unique().tolist()
                errors.append(f"customers: Invalid club_member_status values: {unique_invalid}")

            # Validate fashion news frequency
            invalid_freq = customers[~customers['fashion_news_frequency'].isin(self.valid_fashion_news_freq)]
            if not invalid_freq.empty:
                unique_invalid = invalid_freq['fashion_news_frequency'].unique().tolist()
                errors.append(f"customers: Invalid fashion_news_frequency values: {unique_invalid}")

        # Validate transactions
        if not transactions.empty:
            # Check date range
            start_date = pd.to_datetime('2018-09-20')
            end_date = pd.to_datetime('2018-09-22')
            if not transactions['t_dat'].between(start_date, end_date).all():
                errors.append(f"transactions: Dates outside expected range {start_date} to {end_date}")

            # Validate price range
            if not transactions['price'].between(0, 1).all():
                invalid_prices = transactions[~transactions['price'].between(0, 1)]
                errors.append(f"transactions: {len(invalid_prices)} price values outside 0-1 range")

            # Validate sales channel
            invalid_channels = transactions[~transactions['sales_channel_id'].isin(self.valid_sales_channels)]
            if not invalid_channels.empty:
                errors.append(f"transactions: Invalid sales_channel_id values found")

        # Validate referential integrity
        if not articles.empty and not transactions.empty:
            invalid_articles = ~transactions['article_id'].isin(articles['article_id'])
            if invalid_articles.any():
                errors.append(f"transactions: {invalid_articles.sum()} transactions reference non-existent articles")

        if not customers.empty and not transactions.empty:
            invalid_customers = ~transactions['customer_id'].isin(customers['customer_id'])
            if invalid_customers.any():
                errors.append(f"transactions: {invalid_customers.sum()} transactions reference non-existent customers")

        return errors

    def generate_data_profile(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive data profile."""
        profile = {}

        for name, df in datasets.items():
            if df.empty:
                continue

            profile[name] = {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'null_counts': df.isnull().sum().to_dict(),
                'null_percentages': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'numeric_summary': df.describe().to_dict() if df.select_dtypes(include=['number']).shape[1] > 0 else {}
            }

            # Add categorical summaries
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                profile[name]['categorical_summary'] = {}
                for col in cat_cols[:5]:  # Limit to first 5 categorical columns
                    value_counts = df[col].value_counts()
                    profile[name]['categorical_summary'][col] = {
                        'unique_values': len(value_counts),
                        'top_values': value_counts.head().to_dict(),
                        'most_common': value_counts.index[0] if len(value_counts) > 0 else None
                    }

        return profile

    def run_full_validation(self) -> Dict[str, Any]:
        """Run comprehensive data validation."""
        validation_results = {
            'status': 'PASS',
            'errors': [],
            'warnings': [],
            'profile': {},
            'timestamp': pd.Timestamp.now().isoformat()
        }

        try:
            # Load datasets
            datasets = self.load_datasets()

            # Schema validation
            schema_errors = []
            schema_errors.extend(self.validate_schema(datasets['articles'], self.articles_schema, 'articles'))
            schema_errors.extend(self.validate_schema(datasets['customers'], self.customers_schema, 'customers'))
            schema_errors.extend(self.validate_schema(datasets['transactions'], self.transactions_schema, 'transactions'))

            validation_results['errors'].extend(schema_errors)

            # Business rules validation
            business_errors = self.validate_business_rules(datasets)
            validation_results['errors'].extend(business_errors)

            # Generate profile
            validation_results['profile'] = self.generate_data_profile(datasets)

            # Determine overall status
            if validation_results['errors']:
                validation_results['status'] = 'FAIL'

        except Exception as e:
            validation_results['status'] = 'ERROR'
            validation_results['errors'].append(f"Validation failed with exception: {str(e)}")

        return validation_results


def main():
    """Run data validation as standalone script."""
    validator = DataValidator()
    results = validator.run_full_validation()

    print("🔍 Data Validation Results")
    print("=" * 50)
    print(f"Status: {results['status']}")
    print(f"Timestamp: {results['timestamp']}")

    if results['errors']:
        print(f"\n❌ Errors Found ({len(results['errors'])}):")
        for i, error in enumerate(results['errors'], 1):
            print(f"  {i}. {error}")

    if results['warnings']:
        print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
        for i, warning in enumerate(results['warnings'], 1):
            print(f"  {i}. {warning}")

    if results['status'] == 'PASS':
        print("\n✅ All validation checks passed!")

    # Print basic profile info
    if results['profile']:
        print(f"\n📊 Dataset Profiles:")
        for name, profile in results['profile'].items():
            print(f"  {name}: {profile['shape'][0]:,} rows, {profile['shape'][1]:,} cols, {profile['memory_usage_mb']:.1f}MB")


if __name__ == "__main__":
    main()