"""Unit tests for data validation utilities."""

import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.data_validation import DataValidator


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataValidator()

        # Sample test data
        self.sample_articles = pd.DataFrame({
            'article_id': ['123', '456'],
            'product_code': [123456, 456789],
            'prod_name': ['Test Product 1', 'Test Product 2'],
            'product_type_no': [253, 306],
            'product_type_name': ['Vest top', 'Bra'],
            'product_group_name': ['Garment Upper body', 'Underwear'],
            'graphical_appearance_no': [1010016, 1010016],
            'graphical_appearance_name': ['Solid', 'Solid'],
            'colour_group_code': [9, 10],
            'colour_group_name': ['Black', 'White'],
            'perceived_colour_value_id': [4, 3],
            'perceived_colour_value_name': ['Dark', 'Light'],
            'perceived_colour_master_id': [5, 9],
            'perceived_colour_master_name': ['Black', 'White'],
            'department_no': [1676, 1339],
            'department_name': ['Jersey Basic', 'Clean Lingerie'],
            'index_code': ['A', 'B'],
            'index_name': ['Ladieswear', 'Lingeries/Tights'],
            'index_group_no': [1, 1],
            'index_group_name': ['Ladieswear', 'Ladieswear'],
            'section_no': [16, 61],
            'section_name': ['Womens Everyday Basics', 'Womens Lingerie'],
            'garment_group_no': [1002, 1017],
            'garment_group_name': ['Jersey Basic', 'Under-, Nightwear'],
            'detail_desc': ['Description 1', 'Description 2']
        })

        self.sample_customers = pd.DataFrame({
            'customer_id': ['cust1', 'cust2'],
            'FN': [1.0, None],
            'Active': [1.0, None],
            'club_member_status': ['ACTIVE', 'ACTIVE'],
            'fashion_news_frequency': ['Regularly', 'NONE'],
            'age': [25, 30],
            'postal_code': ['12345', '67890']
        })

        self.sample_transactions = pd.DataFrame({
            't_dat': pd.to_datetime(['2018-09-20', '2018-09-21']),
            'customer_id': ['cust1', 'cust2'],
            'article_id': ['123', '456'],
            'price': [0.05, 0.10],
            'sales_channel_id': [2, 1]
        })

    def test_validate_schema_valid_data(self):
        """Test schema validation with valid data."""
        errors = self.validator.validate_schema(
            self.sample_articles,
            self.validator.articles_schema,
            'articles'
        )
        self.assertEqual(len(errors), 0)

    def test_validate_schema_missing_columns(self):
        """Test schema validation with missing columns."""
        df_missing = self.sample_articles.drop('prod_name', axis=1)
        errors = self.validator.validate_schema(
            df_missing,
            self.validator.articles_schema,
            'articles'
        )
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('Missing columns' in error for error in errors))

    def test_validate_schema_extra_columns(self):
        """Test schema validation with extra columns."""
        df_extra = self.sample_articles.copy()
        df_extra['extra_column'] = 'test'
        errors = self.validator.validate_schema(
            df_extra,
            self.validator.articles_schema,
            'articles'
        )
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('Unexpected columns' in error for error in errors))

    def test_validate_business_rules_valid_data(self):
        """Test business rules validation with valid data."""
        datasets = {
            'articles': self.sample_articles,
            'customers': self.sample_customers,
            'transactions': self.sample_transactions
        }
        errors = self.validator.validate_business_rules(datasets)
        self.assertEqual(len(errors), 0)

    def test_validate_business_rules_invalid_age(self):
        """Test business rules validation with invalid age."""
        invalid_customers = self.sample_customers.copy()
        invalid_customers.loc[0, 'age'] = 150  # Invalid age

        datasets = {
            'articles': self.sample_articles,
            'customers': invalid_customers,
            'transactions': self.sample_transactions
        }
        errors = self.validator.validate_business_rules(datasets)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('Invalid age values' in error for error in errors))

    def test_validate_business_rules_invalid_price(self):
        """Test business rules validation with invalid price."""
        invalid_transactions = self.sample_transactions.copy()
        invalid_transactions.loc[0, 'price'] = 1.5  # Invalid price > 1

        datasets = {
            'articles': self.sample_articles,
            'customers': self.sample_customers,
            'transactions': invalid_transactions
        }
        errors = self.validator.validate_business_rules(datasets)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('price values outside 0-1 range' in error for error in errors))

    def test_validate_business_rules_referential_integrity(self):
        """Test referential integrity validation."""
        invalid_transactions = self.sample_transactions.copy()
        invalid_transactions.loc[0, 'article_id'] = 'nonexistent'  # Invalid article ID

        datasets = {
            'articles': self.sample_articles,
            'customers': self.sample_customers,
            'transactions': invalid_transactions
        }
        errors = self.validator.validate_business_rules(datasets)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('reference non-existent articles' in error for error in errors))

    def test_generate_data_profile(self):
        """Test data profile generation."""
        datasets = {
            'articles': self.sample_articles,
            'customers': self.sample_customers,
            'transactions': self.sample_transactions
        }
        profile = self.validator.generate_data_profile(datasets)

        self.assertIn('articles', profile)
        self.assertIn('customers', profile)
        self.assertIn('transactions', profile)

        # Check profile structure
        for name, data_profile in profile.items():
            self.assertIn('shape', data_profile)
            self.assertIn('memory_usage_mb', data_profile)
            self.assertIn('null_counts', data_profile)
            self.assertIn('null_percentages', data_profile)

    @patch('pandas.read_csv')
    def test_load_datasets(self, mock_read_csv):
        """Test dataset loading."""
        # Mock the read_csv calls
        mock_read_csv.side_effect = [
            self.sample_articles,
            self.sample_customers,
            self.sample_transactions
        ]

        datasets = self.validator.load_datasets()

        self.assertEqual(len(datasets), 3)
        self.assertIn('articles', datasets)
        self.assertIn('customers', datasets)
        self.assertIn('transactions', datasets)
        self.assertEqual(mock_read_csv.call_count, 3)

    def test_validate_sales_channel_valid(self):
        """Test valid sales channel validation."""
        valid_channels = [1, 2]
        transactions = self.sample_transactions.copy()
        transactions['sales_channel_id'] = [1, 2]

        invalid_channels = transactions[~transactions['sales_channel_id'].isin(valid_channels)]
        self.assertEqual(len(invalid_channels), 0)

    def test_validate_sales_channel_invalid(self):
        """Test invalid sales channel validation."""
        invalid_transactions = self.sample_transactions.copy()
        invalid_transactions.loc[0, 'sales_channel_id'] = 3  # Invalid channel

        invalid_channels = invalid_transactions[~invalid_transactions['sales_channel_id'].isin([1, 2])]
        self.assertEqual(len(invalid_channels), 1)


if __name__ == '__main__':
    unittest.main()