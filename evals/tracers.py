from langsmith import Client
from langsmith.run_helpers import traceable
from functools import wraps
from typing import Callable, Optional, List, Union, Any, Dict
import random
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from config.utils import get_logger

logger = get_logger(__name__)
load_dotenv()





def get_langsmith_client():
    try:
        langsmith_client = Client()
        langsmith_client.list_projects(limit=1)  # Simple test call
        logger.info("LangSmith tracing enabled")
    except Exception as e:
        langsmith_client = None
        logger.warning(f"LangSmith tracing disabled: {str(e)}. Continuing without tracing.")
    return langsmith_client


def conditional_trace(
    run_type: str,
    langsmith_client: Union[Client, None],
    name: Optional[str] = None,
    sample_rate: float = 0.5,
    eval_tags: Optional[List[str]] = None,
):
    """
    Decorator that conditionally applies tracing with sampling.
    In case whoever uses it doesn't want to trace, evaluate or bother with LangSmith api key.
    Though getting the api key is super easy. Unlike the rest of the keys, it gotta go into .env file.
    Args:
        run_type: The type of run to be traced
        langsmith_client: The LangSmith client 
        name: The name of the run to be traced
        sample_rate: The sample rate for tracing
        eval_tags: The tags to be added to the run
    """
    def decorator(func: Callable):
        # If tracing is disabled, just return the original function
        if not langsmith_client:
            return func
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # If not sampling for evaluation, just run the function without tracing
            if random.random() > sample_rate:
                return func(*args, **kwargs)
            metadata = {"tags": eval_tags} if eval_tags else {}
            # Create a traced version of the function with our metadata
            traced_func = traceable(
                run_type=run_type, 
                name=name, 
                metadata=metadata
            )(func)
            
            result = traced_func(*args, **kwargs)
            
            return result
        
        return wrapper
    return decorator


def evaluate_hallucination(run):
    """
    Evaluate hallucination by comparing response against retrieved docs
    Returns a tuple of (score, comment)
    """
    try:
        # Get the retrieved docs and response from the run
        inputs = run.inputs
        outputs = run.outputs
        retrieved_docs = inputs['state'].get('retrieved_docs', [])
        messages = [message for message in outputs.get('messages', []) if message.get('type', '') == 'ai']
        
        if not inputs or not outputs:
            return None, "Missing inputs or outputs"
            
        response = messages[-1].get('content', '')
        
        if not retrieved_docs or not response:
            return None, "Missing retrieved docs or response"
            
        # Combine all retrieved doc content
        doc_content = "\n".join([doc.get('content', '') for doc in retrieved_docs])
        
        # Use an LLM to evaluate hallucination
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        eval_prompt = f"""
        You are a hallucination detector. Compare the AI's response against the source documents.
        
        Source Documents:
        {doc_content}
        
        AI Response:
        {response}
        
        Score the response from 0 to 1 where:
        0 = No hallucination, everything is supported by sources
        0.5 = Minor irrelevant expansions that are not supported by sources
        1 = Major hallucinations or completely unsupported claims
        
        Return ONLY a JSON object in this format:
        {{"score": float, "comment": "explanation"}}
        """
        
        result = llm.invoke(eval_prompt)
        evaluation = json.loads(result.content)
        logger.info(f"score: {evaluation['score']}")
        
        return evaluation['score'], evaluation['comment']
        
    except Exception as e:
        logger.error(f"Error evaluating hallucination: {e}")
        return None, f"Evaluation failed: {str(e)}"


# stuff
ls_client = get_langsmith_client()


