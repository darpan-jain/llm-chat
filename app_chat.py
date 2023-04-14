from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from peft import PeftModel
import torch
import transformers
import gradio as gr
import time
import logging
from logging.handlers import RotatingFileHandler

# Set up logging
log_format = '<%(asctime)s> [%(levelname)s] <%(lineno)s> <%(funcName)s> %(message)s'
file_logger = RotatingFileHandler('chat.log', mode='a', maxBytes=500 * 1024 * 1024, backupCount=10)
formatter = logging.Formatter(log_format, datefmt='%m/%d/%Y %I:%M:%S %p')
file_logger.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_logger)

MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = "tloen/alpaca-lora-7b"
device = "cpu"
print(f"Model device = {device}", flush=True)

def load_model():
    logger.info("--"*50)
    logger.info("\n\nLoading model...")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL)
    model = LlamaForCausalLM.from_pretrained(MODEL, device_map={"": device}, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS, device_map={"": device}, torch_dtype=torch.float16)
    model.eval()

    logger.info("Model loaded.")
    return model, tokenizer

def generate_prompt(input):
        return f""" Below A dialog, where User interacts with you - the AI.
        
        ### Instruction: AI is helpful, kind, obedient, honest, and knows its own limits.
        
        ### User: {input}
        
        ### Response:
        """

def eval_prompt(
        model,
        tokenizer,
        input: str,
        temparature = 0.7,
        top_p = 0.75,
        top_k = 40,
        num_beams = 1,
        max_new_tokens = 128,
        **kwargs):

        prompt = generate_prompt(input)
        inputs = tokenizer(prompt, return_tensors = "pt")
        input_ids = inputs["input_ids"]
        generation_config = GenerationConfig(
            temparatue = temparature,
            top_p = top_p,
            top_k = top_k,
            num_beams = num_beams,
            repetition_penalty = 1.17,
            ** kwargs,)

        # with torch.inference_mode():
        with torch.no_grad():
            generation_output = model.generate(
                input_ids = input_ids,
                generation_config = generation_config,
                return_dict_in_generate = True,
                output_scores = True,
                max_new_tokens = max_new_tokens,
            )
            s = generation_output.sequences[0]
            response = tokenizer.decode(s)
            print(f"Bot response: {response.split('### Response:')[-1].strip()}")
            bot_response = response.split("### Response:")[-1].strip()
            return bot_response

def run_app(model, tokenizer):

    logger.info("Starting chat app...\n")

    with gr.Blocks(theme=gr.themes.Soft(), analytics_enabled=True) as chat:
        chatbot = gr.Chatbot(label = "Alpaca Demo")
        msg = gr.Textbox(show_label = False, placeholder = "Enter your text here")
        clear = gr.Button("Clear")

        def user(user_msg, history):
            logger.info("User input received.")
            return "", history + [[user_msg, None]]

        def bot(history):
            logger.info("Processing user input for Alpaca response...")
            last_input = history[-1][0]
            logger.info(f"User input = {last_input}")

            tick = time.time()
            bot_response = eval_prompt(model, tokenizer, last_input)
            logger.info(f"Inference time = {time.time() - tick: .3f} seconds")

            history[-1][1] = bot_response
            logger.info("Response generated and added to history.\n")

            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

        clear.click(lambda: None, None, chatbot, queue=False)

    chat.queue()
    _, local_url, public_url = chat.launch(share=True)
    logger.info(f"Chat app launched. Local url: {local_url} & Public url: {public_url}")


if __name__ == "__main__":

    model, tokenizer = load_model()

    # Run the actual gradio app
    run_app(model, tokenizer)
