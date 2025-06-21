import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os

load_dotenv()  

def upload_file_to_s3(file_name, bucket_name, object_name=None):
    """
    Upload a file to an S3 bucket

    :param file_name: Path to the local file to upload
    :param bucket_name: Name of the target S3 bucket
    :param object_name: S3 object name (key). If not specified, file_name's basename is used
    :return: True if upload succeeded, False otherwise
    """
    # If S3 object_name was not specified, use the file name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Create an S3 client
    session = boto3.Session(
    aws_access_key_id = os.getenv("aws_access_key_id"),
    aws_secret_access_key=os.getenv("aws_secret_access_key"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
    )

    s3_client = session.client('s3')

    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
        print(f"File '{file_name}' uploaded to bucket '{bucket_name}' as '{object_name}'")
        return True
    except ClientError as e:
        print(f"Error uploading file: {e}")
        return False

file_to_upload = 'ivf_sample.pdf'  
target_bucket = 'demo-ai-agent'  
object_key = 'uploads/ivf_sample.pdf'  

upload_file_to_s3(file_to_upload, target_bucket,object_key)
upload_file_to_s3('ivf_sample_2.pdf',target_bucket,'uploads/ivf_sample_2.pdf')

def delete_file_from_s3(bucket_name, object_key):
    """
    Delete a file from an S3 bucket
    
    :param bucket_name: Name of the bucket
    :param object_key: Key (path/filename) of the file in the bucket
    :return: True if deletion succeeded, False otherwise
    """
    # Create S3 client using the session with proper credentials
    session = boto3.Session(
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        region_name=os.getenv("AWS_DEFAULT_REGION")
    )
    s3_client = session.client('s3')  # Use the session to create the client
    
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=object_key)
        print(f"File '{object_key}' deleted from bucket '{bucket_name}'.")
        return True
    except ClientError as e:
        print(f"Error deleting file: {e}")
        return False
    


# response = s3_client.list_buckets()
# for bucket in response['Buckets']:
#     print(bucket['Name'])

# if __name__ == "__main__":
#     bucket = 'demo-ai-agent'  
#     key = 'uploads/1706.03762.pdf'  
    
#     delete_file_from_s3(bucket, key)