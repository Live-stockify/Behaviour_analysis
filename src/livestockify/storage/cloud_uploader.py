"""
Cloud uploader (stub).

Will eventually push detection records to S3, an HTTP API, or both.
For now, this is a placeholder so the architecture is in place.
"""

from __future__ import annotations

from loguru import logger

from livestockify.inference.aggregator import CountRecord


class CloudUploader:
    """
    Placeholder for cloud upload functionality.
    
    TODO:
    - Implement S3 upload (boto3)
    - Implement HTTP API push (requests)
    - Add retry logic with exponential backoff
    - Add local queueing for offline resilience
    """

    def __init__(self, config: dict):
        self.config = config
        logger.info("CloudUploader initialized (stub - no uploads yet)")

    def push(self, record: CountRecord) -> bool:
        """
        Push a single record to the cloud.
        
        Returns:
            True if successful, False otherwise.
        """
        logger.debug(f"[stub] Would push record: {record.timestamp}")
        return True

    def push_batch(self, records: list[CountRecord]) -> int:
        """
        Push multiple records in one go.
        
        Returns:
            Number of records successfully pushed.
        """
        logger.debug(f"[stub] Would push batch of {len(records)} records")
        return len(records)
