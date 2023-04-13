from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from peft import PeftModel
import torch
import transformers
import gradio as gr
import time

MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = "tloen/alpaca-lora-7b"
device = "cpu"
print(f"Model device = {device}", flush=True)

tokenizer = LlamaTokenizer.from_pretrained(MODEL)
model = LlamaForCausalLM.from_pretrained(MODEL, device_map={"": device}, low_cpu_mem_usage=True, )
model = PeftModel.from_pretrained(model, LORA_WEIGHTS, device_map={"": device}, torch_dtype=torch.float16)

model.eval()

def generate_prompt(input, history):
    if not history:
        return f""" Below A dialog, where User interacts with you - the AI.
        
        ### Instruction: AI is helpful, kind, obedient, honest, and knows its own limits.
        
        ### User: {input}
        
        ### Response:
        """

    else:
        return  f"""{history}
        
        ### User: {input}
        
        ### Response:
        """

    # else:
    #     return f""" Below is an instruction that describes a task.  Write a response that appropriately completes the request.
    #
    #     ### Instruction: {instruction}
    #
    #     ### Response:
    #     """

def eval_prompt(
        input: str,
        history = "",
        temparature = 0.7,
        top_p = 0.75,
        top_k = 40,
        num_beams = 1,
        max_new_tokens = 128,
        **kwargs):

        history = generate_prompt(input, history)
        inputs = tokenizer(history, return_tensors = "pt")
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
            # print(response.split('### Response:')[-1].strip())
            bot_response = response.split("### Response:")[-1].strip()
            history += bot_response
            return history, bot_response

# def run_app():
    # g = gr.Interface(
    #     fn = eval_prompt,
    #     inputs = [
    #         gr.components.Textbox(
    #             lines = 2, label = 'Instruction', placeholder= "Enter an instruction here."),
    #         gr.components.Textbox(lines = 2, label = 'Input', placeholder = "Add an input here.")
    #     ],
    #     outputs = [ gr.inputs.Textbox(lines = 5, label = 'Output') ],
    #     title = 'Alpaca Demo'
    # )
    #
    # g.queue(concurrency_count=1)
    # g.launch(share=True, debug=True)

if __name__ == "__main__":
    history = ""

    while True:
        # testing code for readme
        # for instruction in [
            # "Tell me about alpacas.",
            # "Tell me about the president of Mexico in 2019.",
            # "Tell me about the king of France in 2019.",
            # "List all Canadian provinces in alphabetical order.",
            # "Write a Python program that prints the first 10 Fibonacci numbers.",
            # "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
            # "Tell me five words that rhyme with 'shock'.",
            # "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
            # "Count up from 1 to 500.",
        # ]:

            print("Input text here: ", end=' ')
            user_input = input()
            tick = time.time()
            history, response = eval_prompt(user_input, history)
            print(f"Bot: {response}")
            print(f"Present history: {history}")
            print(f"Inference time = {time.time() - tick} seconds")
            print()

    ## Run the actual gradio app
    # run_app()
