import logging
import sys
import time

import win32api
import win32event

logger = logging.getLogger(__name__)

def guard_against_multiple_instances(app_name: str = "LazyToTextLocal"):
    mutex_name = f"{app_name}_SingleInstance"
    
    try:
        mutex_handle = win32event.CreateMutex(None, True, mutex_name)
        
        if win32api.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            logger.info("Another instance detected")
            _exit_to_prevent_duplicate()
        else:
            logger.info("Primary instance acquired mutex")
            # Return the mutex handle so it stays alive until app exits
            return mutex_handle
            
    except Exception as e:
        logger.error(f"Error with single instance check: {e}")
        raise

def _exit_to_prevent_duplicate():
    logger.info("Lazy to text is already running!", extra={'user_message': True})       
    logger.info("This app will close in 3 seconds...", extra={'user_message': True})
    
    for i in range(3, 0, -1):
        time.sleep(1)
    
    logger.info("Goodbye!", extra={'user_message': True})
    sys.exit(0)