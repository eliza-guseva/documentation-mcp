# feedback_processor.py
import time
from datetime import datetime, timedelta
from langsmith import Client
from dotenv import load_dotenv
import os
from config.utils import get_logger, setup_logging
from evals.tracers import evaluate_hallucination

setup_logging()
load_dotenv()

logger = get_logger(__name__)

PROJECT_NAME = os.getenv("LANGSMITH_PROJECT")
logger.info(f"Project name: {PROJECT_NAME}")

def process_missing_feedback():
    """Add feedback to recent runs without feedback"""

    client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    
    # Get runs from your specific project
    runs = list(client.list_runs(
        project_name=PROJECT_NAME,
        run_type="chain",
        start_time=datetime.now() - timedelta(days=1),
    ))
    runs_needing_eval = [
        run for run in runs 
        if not run.feedback_stats and 
        run.extra.get('metadata').get('tags', [])
    ]
    
    # logger.info(f"{runs[0]}")
    
    logger.info(f"Found {len(runs_needing_eval)} runs needing evaluation")
    
    for run in runs_needing_eval:
        logger.info(f"run_id: {run.id}, tags: {run.extra.get('metadata').get('tags', [])}")
        if "hallucination_eval" in run.extra.get('metadata').get('tags', []):
            logger.info(f"Evaluating hallucination for run {run.id}")   
            score, comment = evaluate_hallucination(run)
            logger.info(f"score: {score}, comment: {comment}")
        
            if score is not None:
                feedback = client.create_feedback(
                    run_id=run.id,
                    key="hallucination_score",
                    score=float(score),
                    comment=f"Automatic evaluation: {comment}",
                )
                logger.info(f"feedback: {feedback.id}")
                logger.info(f"Added hallucination feedback to run {run.id}: score={score}, comment={comment}")
            else:
                logger.warning(f"Skipped feedback for run {run.id}: {comment}")
            
        

if __name__ == "__main__":
    logger.info(f"Starting feedback processor for project: {PROJECT_NAME}")
    while True:
        process_missing_feedback()
        time.sleep(300)