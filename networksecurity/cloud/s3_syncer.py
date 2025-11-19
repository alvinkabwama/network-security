import os  # Provides a way to interact with the operating system (used here to run shell commands)

class S3Sync:
    # Syncs a local folder *to* an S3 bucket using the AWS CLI 'aws s3 sync' command
    def sync_folder_to_s3(self,folder,aws_bucket_url):
        # Build the AWS CLI sync command: copies local folder contents to the specified S3 bucket URL
        command = f"aws s3 sync {folder} {aws_bucket_url} "
        # Execute the command in the system shell
        os.system(command)

    # Syncs a folder *from* an S3 bucket to a local folder using the AWS CLI 'aws s3 sync' command
    def sync_folder_from_s3(self,folder,aws_bucket_url):
        # Build the AWS CLI sync command: copies contents from the S3 bucket URL to the local folder
        command = f"aws s3 sync  {aws_bucket_url} {folder} "
        # Execute the command in the system shell
        os.system(command)
