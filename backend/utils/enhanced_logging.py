# backend/utils/enhanced_logging.py
"""
Enhanced Logging Module
======================
This module provides improved logging functionality for the ASTER processing system,
including detailed logging for band combinations and processing operations.

Author: GIS Remote Sensing
"""

import os
import sys
import logging
import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Dict, Optional, Union, List

class ColorFormatter(logging.Formatter):
    """Custom formatter that adds colors to terminal output"""
    
    COLORS = {
        'DEBUG': '\033[94m',      # Blue
        'INFO': '\033[92m',       # Green
        'WARNING': '\033[93m',    # Yellow
        'ERROR': '\033[91m',      # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format the log record with color"""
        # Windows console doesn't support ANSI color codes by default
        if os.name == 'nt' and not 'ANSICON' in os.environ:
            return super().format(record)
            
        levelname = record.levelname
        message = super().format(record)
        
        # Add color to the levelname in the message
        if levelname in self.COLORS:
            color_start = self.COLORS[levelname]
            color_end = self.COLORS['RESET']
            message = message.replace(levelname, f"{color_start}{levelname}{color_end}", 1)
            
        return message

def setup_logging(log_dir: Union[str, Path] = None, 
                 log_level: int = logging.INFO, 
                 log_to_console: bool = True,
                 log_to_file: bool = True,
                 module_name: str = None) -> logging.Logger:
    """
    Set up enhanced logging for the application
    
    Parameters:
    -----------
    log_dir : Union[str, Path], optional
        Directory to store log files
    log_level : int, optional
        Logging level (default: logging.INFO)
    log_to_console : bool, optional
        Whether to log to console (default: True)
    log_to_file : bool, optional
        Whether to log to file (default: True)
    module_name : str, optional
        Name of the module for the logger
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create logger
    if module_name:
        logger = logging.getLogger(module_name)
    else:
        logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set level
    logger.setLevel(log_level)
    
    # Create formatters
    console_formatter = ColorFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handlers if requested
    if log_to_file:
        if not log_dir:
            log_dir = Path.cwd() / 'logs'
        else:
            log_dir = Path(log_dir)
            
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use current date for log file name
        date_str = datetime.datetime.now().strftime('%Y%m%d')
        module_prefix = f"{module_name}_" if module_name else ""
        log_file = log_dir / f"{module_prefix}{date_str}.log"
        
        # Create file handler with rotation
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Create daily rotating handler for long-term logging
        daily_log_file = log_dir / f"{module_prefix}daily_rotating.log"
        daily_handler = TimedRotatingFileHandler(
            daily_log_file,
            when='midnight',
            interval=1,
            backupCount=30  # Keep 30 days of logs
        )
        daily_handler.setFormatter(file_formatter)
        logger.addHandler(daily_handler)
    
    logger.info(f"Logging initialized at level {logging.getLevelName(log_level)}")
    
    return logger

def log_processing_operation(logger: logging.Logger, 
                           operation: str, 
                           scene_id: str = None, 
                           parameters: Dict = None,
                           start_time: datetime.datetime = None) -> datetime.datetime:
    """
    Log the start or end of a processing operation
    
    Parameters:
    -----------
    logger : logging.Logger
        Logger to use for logging
    operation : str
        Name of the operation
    scene_id : str, optional
        ID of the scene being processed
    parameters : Dict, optional
        Parameters for the operation (for start logs)
    start_time : datetime.datetime, optional
        Start time of the operation (for end logs)
        
    Returns:
    --------
    datetime.datetime
        Current time (for tracking operation duration)
    """
    current_time = datetime.datetime.now()
    
    if start_time is None:
        # This is a start log
        scene_str = f" for scene {scene_id}" if scene_id else ""
        logger.info(f"Starting operation: {operation}{scene_str}")
        
        if parameters:
            param_str = "\n".join([f"    {k}: {v}" for k, v in parameters.items()])
            logger.info(f"Operation parameters:\n{param_str}")
            
        return current_time
    else:
        # This is an end log
        duration = current_time - start_time
        scene_str = f" for scene {scene_id}" if scene_id else ""
        logger.info(f"Completed operation: {operation}{scene_str}")
        logger.info(f"Operation duration: {duration}")
        return current_time

def log_exception(logger: logging.Logger, 
                operation: str, 
                exception: Exception, 
                scene_id: str = None) -> None:
    """
    Log an exception that occurred during processing
    
    Parameters:
    -----------
    logger : logging.Logger
        Logger to use for logging
    operation : str
        Name of the operation during which the exception occurred
    exception : Exception
        The exception that occurred
    scene_id : str, optional
        ID of the scene being processed
    """
    scene_str = f" for scene {scene_id}" if scene_id else ""
    logger.error(f"Error during operation: {operation}{scene_str}")
    logger.error(f"Exception type: {type(exception).__name__}")
    logger.error(f"Exception message: {str(exception)}")
    logger.exception("Stack trace:")

def get_performance_stats(logger: logging.Logger, 
                        scene_id: str,
                        processing_steps: List[str],
                        durations: List[float]) -> None:
    """
    Log performance statistics for processing steps
    
    Parameters:
    -----------
    logger : logging.Logger
        Logger to use for logging
    scene_id : str
        ID of the scene being processed
    processing_steps : List[str]
        Names of the processing steps
    durations : List[float]
        Durations of the processing steps in seconds
    """
    if len(processing_steps) != len(durations):
        logger.error("Mismatch between processing steps and durations")
        return
    
    logger.info(f"Performance statistics for scene {scene_id}:")
    
    total_duration = sum(durations)
    for step, duration in zip(processing_steps, durations):
        percentage = (duration / total_duration) * 100 if total_duration > 0 else 0
        logger.info(f"  {step}: {duration:.2f} seconds ({percentage:.1f}%)")
    
    logger.info(f"Total processing time: {total_duration:.2f} seconds")

def create_summary_log(log_dir: Union[str, Path], 
                     summary_file: str = "processing_summary.log") -> None:
    """
    Create a summary log of all processing operations
    
    Parameters:
    -----------
    log_dir : Union[str, Path]
        Directory containing log files
    summary_file : str, optional
        Name of the summary file to create
    """
    log_dir = Path(log_dir)
    summary_path = log_dir / summary_file
    
    # Define patterns to match in log files
    start_pattern = "Starting operation:"
    complete_pattern = "Completed operation:"
    error_pattern = "Error during operation:"
    
    operations = {}
    
    # Process all log files
    log_files = list(log_dir.glob("*.log"))
    
    with open(summary_path, 'w') as summary:
        summary.write("ASTER Processing Summary\n")
        summary.write("======================\n\n")
        
        summary.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary.write(f"Log files analyzed: {len(log_files)}\n\n")
        
        for log_file in log_files:
            with open(log_file, 'r') as f:
                for line in f:
                    # Extract timestamp and message
                    parts = line.split(' - ', 3)
                    if len(parts) < 4:
                        continue
                    
                    timestamp = parts[0]
                    level = parts[2]
                    message = parts[3].strip()
                    
                    # Track operations
                    if start_pattern in message:
                        op_name = message.split(start_pattern, 1)[1].strip()
                        if op_name not in operations:
                            operations[op_name] = {'starts': 0, 'completes': 0, 'errors': 0}
                        operations[op_name]['starts'] += 1
                    
                    elif complete_pattern in message:
                        op_name = message.split(complete_pattern, 1)[1].strip()
                        if op_name not in operations:
                            operations[op_name] = {'starts': 0, 'completes': 0, 'errors': 0}
                        operations[op_name]['completes'] += 1
                    
                    elif error_pattern in message:
                        op_name = message.split(error_pattern, 1)[1].strip()
                        if op_name not in operations:
                            operations[op_name] = {'starts': 0, 'completes': 0, 'errors': 0}
                        operations[op_name]['errors'] += 1
        
        # Write operation summary
        summary.write("Operation Summary:\n")
        summary.write("-----------------\n")
        
        for op_name, stats in operations.items():
            summary.write(f"\n{op_name}:\n")
            summary.write(f"  Started: {stats['starts']} times\n")
            summary.write(f"  Completed: {stats['completes']} times\n")
            summary.write(f"  Errors: {stats['errors']} times\n")
            
            if stats['starts'] > stats['completes'] + stats['errors']:
                summary.write(f"  Warning: {stats['starts'] - stats['completes'] - stats['errors']} operations may still be running or were interrupted\n")

if __name__ == "__main__":
    """Example usage of the enhanced logging module"""
    # Set up logging
    logger = setup_logging(log_dir="./logs", module_name="aster_processing")
    
    # Log some test messages
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Log processing operation
    start_time = log_processing_operation(
        logger, 
        "test_operation", 
        scene_id="TEST001", 
        parameters={"param1": "value1", "param2": 42}
    )
    
    # Simulate processing
    import time
    time.sleep(1)
    
    # Log completion
    log_processing_operation(logger, "test_operation", scene_id="TEST001", start_time=start_time)
    
    # Log exception
    try:
        raise ValueError("Test exception")
    except Exception as e:
        log_exception(logger, "test_operation", e, scene_id="TEST001")
    
    # Log performance stats
    get_performance_stats(
        logger,
        "TEST001",
        ["step1", "step2", "step3"],
        [1.5, 2.7, 0.8]
    )
    
    # Create summary log
    create_summary_log("./logs")