import boto3
from botocore.exceptions import ClientError
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class S3FileHandler:
    def __init__(self, access_key, secret_key, bucket_name, base_prefix):
        self.aws_access_key = access_key
        self.aws_secret_key = secret_key
        self.bucket_name = bucket_name
        self.base_prefix = base_prefix

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key
        )

    def check_file_exists(self, s3_key):
        """Check if a file exists in S3 using head_object"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True, s3_key
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False, None
            else:
                self.logger.error(f"❌ Error checking file {s3_key}: {str(e)}")
                return False, None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error checking file {s3_key}: {str(e)}")
            return False, None

    def list_files_by_hotel_user(self, hotel_code, user_id, chart_ids=None):
        """
        Checks S3 for all valid chart files matching pattern: live_dataset_<chartid>_<hotelcode>_<userid>.json
        using parallel head_object checks. If chart_ids is provided, only those chart IDs are checked.
        """
        # Create a cache key
        cache_key = f"{hotel_code}_{user_id}_{str(chart_ids)}"
        
        # Check if we have cached results
        if hasattr(self, '_file_cache') and cache_key in self._file_cache:
            return self._file_cache[cache_key]
        
        self.logger.info(f"🔍 Searching for files in S3 for hotel {hotel_code}...")
        found_files = []
        
        try:
            # Use provided chart_ids or default to 1-80
            if chart_ids is None:
                chart_ids = range(1, 81)
            self.logger.info(f"Checking these chart IDs: {chart_ids}")
            file_keys = [
                f"{self.base_prefix}live_dataset_{chart_id}_{hotel_code}_{user_id}.json"
                for chart_id in chart_ids
            ]
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks
                future_to_key = {
                    executor.submit(self.check_file_exists, key): key 
                    for key in file_keys
                }
                
                # Process results as they complete
                for future in as_completed(future_to_key):
                    exists, key = future.result()
                    if exists:
                        found_files.append(key)
                        self.logger.debug(f"✅ Found file: {key}")
            
            # Cache the results
            if not hasattr(self, '_file_cache'):
                self._file_cache = {}
            self._file_cache[cache_key] = found_files
            
            self.logger.info(f"✅ Found {len(found_files)} files for hotel {hotel_code}")
            return found_files
            
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {str(e)}")
            return []

    def read_file_content(self, file_key):
        """Read file content from S3 with retry mechanism"""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
                return response['Body'].read().decode('utf-8')
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    return None
                elif attempt < max_retries - 1:
                    self.logger.warning(f"Retry {attempt + 1}/{max_retries} for {file_key}")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"❌ Error reading file {file_key}: {str(e)}")
                    return None
            except Exception as e:
                self.logger.error(f"❌ Unexpected error reading file {file_key}: {str(e)}")
                return None

    def validate_inputs(self, hotel_code, user_id):
        """Validate hotel code and user ID"""
        if not hotel_code or not user_id:
            return False, "Hotel code and User ID are required."
        if len(hotel_code) < 3:
            return False, "Hotel code should be at least 3 characters."
        if len(user_id) < 4:
            return False, "User ID should be at least 4 characters."
        return True, "Valid inputs." 