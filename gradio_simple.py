import logging
import gradio as gr

from agent.agents import (
    create_multi_query_agent_executor, 
    stream_agent_output,
    StateCapture
)
from config.utils import setup_logging, get_logger
from config.config import ConfigManager


setup_logging(level=logging.INFO)
logger = get_logger(__name__)
logger.warning("Starting the application...")
logger.info("Starting the application...")
config = ConfigManager()
   
state_capture = StateCapture()
agent_executor = create_multi_query_agent_executor()


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=420)
    # Text input with stop button on right
    with gr.Row():
        with gr.Column(scale=20):
            msg = gr.Textbox(label="Message", placeholder="Ask something...")
            
        with gr.Column(scale=1, min_width=40):
            stop_btn = gr.Button("⏹️", size="sm", variant="stop")
    
    # Buttons below the text input        
    with gr.Row():
        send_btn = gr.Button("Send", size="md", variant="primary", scale=3)
        clear = gr.Button("Clear", size="md", scale=1)
    
    def respond(message, chat_history):
        # Add user message to chat history
        chat_history.append((message, ""))
        yield chat_history
        
        # Stream the agent's response
        bot_message = ""
        for chunk in stream_agent_output(agent_executor, message, state_capture):
            bot_message += chunk
            # Update just the last bot message
            chat_history[-1] = (message, bot_message)
            yield chat_history
    
    # Set up events with cancellation
    msg_event = msg.submit(respond, [msg, chatbot], [chatbot])
    send_event = send_btn.click(respond, [msg, chatbot], [chatbot])
    
    # Clear button
    clear.click(lambda: None, None, chatbot, queue=False)
    
    # Stop button cancels both the message submit and send button events
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[msg_event, send_event])

demo.launch(inline=False, share=False)