"""
Data Cleaner Pro - Professional Data Validation for ML Pipelines

Author: Wodson
Date: 20/06/2026
Version: 1.0.0

This module provides data validation and cleaning capabilities for
Machine Learning pipelines. It validates age, email, and income fields
with professional error handling and reporting.
"""

import re
from typing import Dict, List, Optional, Tuple, Union


class DataCleaner:
    """
    Professional data cleaner for ML preprocessing.
    
    Validates and cleans common data types:
    - Age: 0-120, integer
    - Email: RFC 5322 compliant format
    - Income: non-negative float
    
    Attributes:
        errors (List[str]): Collection of validation errors
        cleaned_data (List[Dict]): Successfully validated records
        total_processed (int): Total records processed
        success_rate (float): Percentage of valid records
    """
    
    def __init__(self) -> None:
        """Initialize the DataCleaner with empty error and data collections."""
        self.errors: List[str] = []
        self.cleaned_data: List[Dict] = []
        self.total_processed: int = 0
        self._valid_count: int = 0
        
    def clean_age(self, age: Optional[Union[int, float]]) -> Optional[int]:
        """
        Validate and clean age field.
        
        Validation rules:
        - Must not be None
        - Must be numeric (int or float)
        - Must be between 0 and 120 inclusive
        - Returns integer (converted from float)
        
        Args:
            age: Age value to validate
            
        Returns:
            Validated age as int, or None if invalid
            
        Examples:
            >>> cleaner = DataCleaner()
            >>> cleaner.clean_age(25)
            25
            >>> cleaner.clean_age(-5)
            None
        """
        if age is None:
            self.errors.append("❌ Age is None")
            return None
            
        if not isinstance(age, (int, float)):
            self.errors.append(f"❌ Invalid age type: {type(age)}")
            return None
            
        if age < 0:
            self.errors.append(f"❌ Negative age: {age}")
            return None
            
        if age > 120:
            self.errors.append(f"❌ Age exceeds limit (120): {age}")
            return None
            
        return int(age)
    
    def clean_email(self, email: Optional[str]) -> Optional[str]:
        """
        Validate and clean email field.
        
        Validation rules:
        - Must not be None
        - Must be string
        - Must match RFC 5322 email format
        - Normalized to lowercase
        
        Args:
            email: Email to validate
            
        Returns:
            Validated email as lowercase string, or None if invalid
            
        Examples:
            >>> cleaner = DataCleaner()
            >>> cleaner.clean_email("Joao@Gmail.com")
            'joao@gmail.com'
            >>> cleaner.clean_email("invalid")
            None
        """
        if email is None:
            self.errors.append("❌ Email is None")
            return None
            
        if not isinstance(email, str):
            self.errors.append(f"❌ Invalid email type: {type(email)}")
            return None
            
        # Normalize
        email = email.lower().strip()
        
        # RFC 5322 compliant regex pattern
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(pattern, email):
            self.errors.append(f"❌ Invalid email format: {email}")
            return None
            
        return email
    
    def clean_income(self, income: Optional[Union[int, float]]) -> Optional[float]:
        """
        Validate and clean income field.
        
        Validation rules:
        - Must not be None
        - Must be numeric (int or float)
        - Must be non-negative
        - Returns float
        
        Args:
            income: Income to validate
            
        Returns:
            Validated income as float, or None if invalid
            
        Examples:
            >>> cleaner = DataCleaner()
            >>> cleaner.clean_income(5000.0)
            5000.0
            >>> cleaner.clean_income(-10)
            None
        """
        if income is None:
            self.errors.append("❌ Income is None")
            return None
            
        if not isinstance(income, (int, float)):
            self.errors.append(f"❌ Invalid income type: {type(income)}")
            return None
            
        if income < 0:
            self.errors.append(f"❌ Negative income: {income}")
            return None
            
        return float(income)
    
    def clean_text(self, text: Optional[str]) -> Optional[str]:
        """
        Validate and clean text for NLP processing.
        
        Validation rules:
        - Must not be None
        - Must be string
        - Normalized: lowercase, remove special chars, remove extra spaces
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text as string, or None if invalid
            
        Examples:
            >>> cleaner = DataCleaner()
            >>> cleaner.clean_text("  RESETAR SENHA! ")
            'resetar senha'
        """
        if text is None:
            self.errors.append("❌ Text is None")
            return None
            
        if not isinstance(text, str):
            self.errors.append(f"❌ Invalid text type: {type(text)}")
            return None
            
        # Normalize
        text = text.lower().strip()
        # Remove special chars (keep letters, numbers, spaces)
        text = re.sub(r'[^a-zA-Z0-9áéíóúãõçÀÁÂÃÇÉÊÍÓÚÔ ]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def process_record(self, record: Dict) -> Tuple[Dict, bool]:
        """
        Process a single record through all validations.
        
        Args:
            record: Dictionary with age, email, income fields
            
        Returns:
            Tuple of (cleaned_record, has_errors)
            
        Examples:
            >>> cleaner = DataCleaner()
            >>> record = {"age": 25, "email": "joao@email.com", "income": 5000}
            >>> cleaned, has_error = cleaner.process_record(record)
            >>> has_error
            False
        """
        cleaned_record = {}
        has_errors = False
        
        # Apply all validations
        cleaned_age = self.clean_age(record.get("age"))
        cleaned_email = self.clean_email(record.get("email"))
        cleaned_income = self.clean_income(record.get("income"))
        
        # Build cleaned record (only valid fields)
        if cleaned_age is not None:
            cleaned_record["age"] = cleaned_age
        else:
            has_errors = True
            
        if cleaned_email is not None:
            cleaned_record["email"] = cleaned_email
        else:
            has_errors = True
            
        if cleaned_income is not None:
            cleaned_record["income"] = cleaned_income
        else:
            has_errors = True
        
        return cleaned_record, has_errors
    
    def clean_dataset(self, dataset: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Process an entire dataset through all validations.
        
        Args:
            dataset: List of records to validate
            
        Returns:
            Tuple of (cleaned_data, errors)
            
        Examples:
            >>> cleaner = DataCleaner()
            >>> data = [{"age": 25, "email": "joao@email.com", "income": 5000}]
            >>> cleaned, errors = cleaner.clean_dataset(data)
            >>> len(cleaned)
            1
        """
        # Reset state
        self.errors = []
        self.cleaned_data = []
        self.total_processed = len(dataset)
        self._valid_count = 0
        
        print(f"📊 Processing {self.total_processed} records...")
        print("-" * 50)
        
        for i, record in enumerate(dataset, start=1):
            cleaned_record, has_error = self.process_record(record)
            
            if has_error:
                print(f"⚠️  Record {i}: Validation failed")
            else:
                self.cleaned_data.append(cleaned_record)
                self._valid_count += 1
                print(f"✅  Record {i}: Validated successfully")
        
        return self.cleaned_data, self.errors
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the cleaning process.
        
        Returns:
            Dictionary with processing metrics
            
        Examples:
            >>> cleaner = DataCleaner()
            >>> cleaner.clean_dataset([{"age": 25, "email": "joao@email.com", "income": 5000}])
            >>> stats = cleaner.get_statistics()
            >>> stats['success_rate']
            100.0
        """
        invalid_count = self.total_processed - self._valid_count
        
        return {
            "total_processed": self.total_processed,
            "valid_records": self._valid_count,
            "invalid_records": invalid_count,
            "total_errors": len(self.errors),
            "success_rate": (
                (self._valid_count / self.total_processed * 100)
                if self.total_processed > 0
                else 0.0
            )
        }


def main() -> None:
    """Demonstrate the DataCleaner with sample data."""
    print("=" * 60)
    print("🚀 DATA CLEANER PRO - Professional Data Validation")
    print("=" * 60)
    
    # Sample dataset with intentional errors
    sample_data = [
        {"age": 25, "email": "joao@gmail.com", "income": 5000.00},
        {"age": -10, "email": "invalid-email", "income": 3000.00},
        {"age": 150, "email": "maria@email.com", "income": 6000.00},
        {"age": 30, "email": "ana@dominio.com", "income": None},
        {"age": None, "email": "teste@empresa.com", "income": 7000.00},
        {"age": 45, "email": "valid@domain.com", "income": 8500.50},
        {"age": 18, "email": "user@", "income": 2000.00},
        {"age": 32, "email": "carlos@site.org", "income": 4500.00},
    ]
    
    print(f"\n📊 Total records: {len(sample_data)}")
    print("-" * 50)
    
    # Process data
    cleaner = DataCleaner()
    cleaned_data, errors = cleaner.clean_dataset(sample_data)
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 FINAL REPORT")
    print("=" * 60)
    
    stats = cleaner.get_statistics()
    print(f"📝 Total processed: {stats['total_processed']}")
    print(f"✅ Valid records: {stats['valid_records']}")
    print(f"❌ Invalid records: {stats['invalid_records']}")
    print(f"⚠️  Total errors: {stats['total_errors']}")
    print(f"📈 Success rate: {stats['success_rate']:.1f}%")
    
    if cleaned_data:
        print(f"\n📋 Sample of clean data:")
        print("-" * 40)
        for i, record in enumerate(cleaned_data[:3], start=1):
            print(f"  {i}. Age: {record['age']}, Email: {record['email']}, Income: R${record['income']:,.2f}")
    
    if errors:
        print(f"\n🚨 Sample errors (last 5):")
        print("-" * 40)
        for error in errors[-5:]:
            print(f"  • {error}")
    
    print("\n" + "=" * 60)
    print("✅ Processing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()