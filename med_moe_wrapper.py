import sys
import argparse
from IPython.display import display, Image
import os

# Import the necessary modules from the original CLI script
# You'll need to adapt this based on how the original CLI is structured
sys.path.append(os.path.abspath('.'))  # Ensure the project root is in path
from moellava.serve.cli import main as cli_main
from moellava.serve.cli import load_model, process_image, generate_response
# Note: The above imports are hypothetical - you'll need to check the actual structure

def run_med_moe(model_path, image_path, load_4bit=False, prompt=None):
    """
    Run the Med-MoE model on an image with an optional prompt.
    
    Args:
        model_path (str): Path to the model
        image_path (str): Path to the image
        load_4bit (bool): Whether to load in 4-bit mode
        prompt (str): The prompt to use with the image
        
    Returns:
        str: The model's response
    """
    # Load the model
    model, tokenizer, image_processor, context_len = load_model(
        model_path=model_path,
        load_4bit=load_4bit
    )
    
    # Process the image
    image_tensor = process_image(image_path, image_processor)
    
    # Display the image in the notebook
    display(Image(filename=image_path))
    
    # If no prompt is provided, get it from the user
    if prompt is None:
        prompt = input("Enter your prompt: ")
    
    # Generate the response
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        image_tensor=image_tensor,
        prompt=prompt,
        context_len=context_len
    )
    
    return response

if __name__ == "__main__":
    # This allows the script to be run directly if needed
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--prompt", type=str)
    
    args = parser.parse_args()
    
    response = run_med_moe(
        args.model_path,
        args.image_file,
        args.load_4bit,
        args.prompt
    )
    
    print(response) 
