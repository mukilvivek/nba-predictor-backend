#!/usr/bin/env python3
import time
import schedule
from train_model import main as train_model

def update_job():
    """Run model training job."""
    print("\nChecking for model updates...")
    train_model()

def main():
    # Schedule the job to run daily at midnight
    schedule.every().day.at("00:00").do(update_job)
    
    # Also run immediately on start
    update_job()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main() 