# feedback_processor.py
import time
from datetime import datetime, timedelta
from langsmith import Client
from dotenv import load_dotenv
import os
from config.utils import get_logger, setup_logging
from evals.tracers import evaluate, hallucination_classiffier_prompt, quality_classifier_prompt

setup_logging()
load_dotenv()

logger = get_logger(__name__)

PROJECT_NAME = os.getenv("LANGSMITH_PROJECT")
logger.info(f"Project name: {PROJECT_NAME}")


def process_missing_feedback():
    """Add feedback to recent runs without feedback"""

    client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    
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
    
    
    logger.info(f"Found {len(runs_needing_eval)} runs needing evaluation")
    
    for run in runs_needing_eval:
        logger.info(f"run_id: {run.id}, tags: {run.extra.get('metadata').get('tags', [])}")
        logger.info(f"run.extra: {run.extra}")
        if "hallucination_eval" in run.extra.get('metadata').get('tags', []):
            logger.info(f"Evaluating hallucination for run {run.id}")   
            evaluation = evaluate(run, hallucination_classiffier_prompt)
            
        
            if evaluation.get('expansion_score') is not None:
                feedback = client.create_feedback(
                    run_id=run.id,
                    key="expansion_score",
                    score=float(evaluation.get('expansion_score')),
                    comment=f"Automatic evaluation: {evaluation.get('comment')}",
                )
            if evaluation.get('contradiction_score') is not None:
                feedback = client.create_feedback(
                    run_id=run.id,
                    key="contradiction_score",
                    score=float(evaluation.get('contradiction_score')),
                    comment=f"Automatic evaluation: {evaluation.get('comment')}",
                )
                logger.info(f"feedback: {feedback.id}")
            else:
                logger.warning(f"Skipped feedback for run {run.id}: {evaluation.get('comment')}")
        if "quality_eval" in run.extra.get('metadata').get('tags', []):
            logger.info(f"Evaluating quality for run {run.id}")
            evaluation = evaluate(run, quality_classifier_prompt)
            
            if evaluation.get('answer_score') is not None:
                feedback = client.create_feedback(
                    run_id=run.id,
                    key="answer_score",
                    score=float(evaluation.get('answer_score')),
                )
                
            if evaluation.get('unnecessary_info_score') is not None:
                feedback = client.create_feedback(
                    run_id=run.id,
                    key="unnecessary_info_score",
                    score=float(evaluation.get('unnecessary_info_score')),
                )
                
            if evaluation.get('helpful_score') is not None:
                feedback = client.create_feedback(
                    run_id=run.id,
                    key="helpful_score",
                    score=float(evaluation.get('helpful_score')),
                )
                
            if evaluation.get('answer_score') is None and evaluation.get('unnecessary_info_score') is None and evaluation.get('helpful_score') is None:
                logger.warning(f"Skipped feedback for run {run.id}: {evaluation.get('comment')}")
            else:
                logger.info(f"Added quality feedback to run {run.id}")
        

if __name__ == "__main__":
    logger.info(f"Starting feedback processor for project: {PROJECT_NAME}")
    while True:
        try:
            process_missing_feedback()
        except AttributeError as e:
            # sometimes OAI returns nonsense formatting (even with t=0 it's inconsistent)
            logger.error(f"Error processing feedback: {e}")
            time.sleep(60)
        time.sleep(60)