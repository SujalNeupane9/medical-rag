import boto3
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure AWS credentials
os.environ['aws_access_key_id'] = os.getenv("aws_access_key_id")
os.environ['aws_secret_access_key'] = os.getenv("aws_secret_access_key")
os.environ['AWS_DEFAULT_REGION'] = os.getenv("AWS_DEFAULT_REGION")

def explore_s3_bucket(bucket_name="demo-ai-agent"):
    """Explore the S3 bucket structure and find all files"""
    try:
        # Create S3 client
        s3_client = boto3.client('s3')
        
        print(f"🔍 Exploring S3 bucket: {bucket_name}")
        print("=" * 50)
        
        # List all objects in the bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' not in response:
            print("❌ No objects found in the bucket")
            return
        
        # Group objects by prefix (folder structure)
        folders = {}
        pdf_files = []
        all_files = []
        
        for obj in response['Contents']:
            key = obj['Key']
            size = obj['Size']
            modified = obj['LastModified']
            
            all_files.append({
                'key': key,
                'size': size,
                'modified': modified
            })
            
            # Check if it's a PDF
            if key.lower().endswith('.pdf'):
                pdf_files.append(key)
            
            # Extract folder structure
            if '/' in key:
                folder = '/'.join(key.split('/')[:-1]) + '/'
                if folder not in folders:
                    folders[folder] = []
                folders[folder].append(key)
            else:
                if 'root' not in folders:
                    folders['root'] = []
                folders['root'].append(key)
        
        # Display folder structure
        print(f"📁 FOLDER STRUCTURE:")
        print("-" * 30)
        for folder, files in sorted(folders.items()):
            print(f"📂 {folder}")
            for file in files:
                file_info = next(f for f in all_files if f['key'] == file)
                size_mb = file_info['size'] / (1024 * 1024)
                print(f"   📄 {file.split('/')[-1]} ({size_mb:.2f} MB)")
        
        print(f"\n📋 SUMMARY:")
        print("-" * 30)
        print(f"Total objects: {len(all_files)}")
        print(f"Total PDF files: {len(pdf_files)}")
        print(f"Total folders: {len([f for f in folders.keys() if f != 'root'])}")
        
        if pdf_files:
            print(f"\n🔍 PDF FILES FOUND:")
            print("-" * 30)
            for pdf in pdf_files:
                file_info = next(f for f in all_files if f['key'] == pdf)
                size_mb = file_info['size'] / (1024 * 1024)
                modified_str = file_info['modified'].strftime('%Y-%m-%d %H:%M:%S')
                print(f"📄 {pdf}")
                print(f"   Size: {size_mb:.2f} MB")
                print(f"   Modified: {modified_str}")
                
                # Determine prefix
                if '/' in pdf:
                    prefix = '/'.join(pdf.split('/')[:-1]) + '/'
                    print(f"   Prefix: {prefix}")
                else:
                    print(f"   Prefix: (root level)")
                print()
        
        return folders, pdf_files, all_files
        
    except Exception as e:
        print(f"❌ Error exploring bucket: {str(e)}")
        return None, None, None

def check_specific_prefixes(bucket_name="demo-ai-agent", prefixes=["uploads/", "medical-docs/"]):
    """Check specific prefixes for files"""
    try:
        s3_client = boto3.client('s3')
        
        print(f"\n🔍 Checking specific prefixes in bucket: {bucket_name}")
        print("=" * 50)
        
        for prefix in prefixes:
            print(f"\n📂 Checking prefix: '{prefix}'")
            print("-" * 30)
            
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                print(f"✅ Found {len(response['Contents'])} objects with prefix '{prefix}':")
                for obj in response['Contents']:
                    size_mb = obj['Size'] / (1024 * 1024)
                    print(f"   📄 {obj['Key']} ({size_mb:.2f} MB)")
            else:
                print(f"❌ No objects found with prefix '{prefix}'")
    
    except Exception as e:
        print(f"❌ Error checking prefixes: {str(e)}")

if __name__ == "__main__":
    # Explore the entire bucket
    folders, pdf_files, all_files = explore_s3_bucket()
    
    # Check specific prefixes
    check_specific_prefixes()
    
    # Provide recommendations
    if pdf_files:
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 30)
        print("Based on the files found, you should use these prefixes in your PDFProcessor:")
        
        unique_prefixes = set()
        for pdf in pdf_files:
            if '/' in pdf:
                prefix = '/'.join(pdf.split('/')[:-1]) + '/'
                unique_prefixes.add(prefix)
        
        for prefix in sorted(unique_prefixes):
            print(f"   processor = PDFProcessor(s3_prefix='{prefix}')")
        
        if not unique_prefixes:
            print(f"   processor = PDFProcessor(s3_prefix='')  # Files are at root level")