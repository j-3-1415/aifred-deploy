import textwrap
import torch
import argparse

from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline


def get_prompt(human_prompt):
    prompt_template=f"### Human: {human_prompt} \n### Assistant:"
    return prompt_template


def remove_human_text(text):
    return text.split("### Human:", 1)[0]


def parse_text(data):
    for item in data:
        text = item["generated_text"]
        assistant_text_index = text.find("### Assistant:")
        if assistant_text_index != -1:
            assistant_text = text[assistant_text_index+len("### Assistant:"):].strip()
            assistant_text = remove_human_text(assistant_text)
            wrapped_text = textwrap.fill(assistant_text, width=100)
            return wrapped_text


def run_inference(
        input_txt,
        output_file=None
):
    tokenizer = LlamaTokenizer.from_pretrained("TheBloke/wizardLM-7B-HF")

    base_model = LlamaForCausalLM.from_pretrained(
        "TheBloke/wizardLM-7B-HF",
        load_in_8bit=True,
        device_map='auto',
    )

    pipe = pipeline(
        "text-generation",
        model=base_model, 
        tokenizer=tokenizer, 
        max_length=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    raw_output = pipe(get_prompt(input_txt))
    output_txt = parse_text(raw_output)

    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(output_txt)
    
    print(output_txt)
    

def main():
    parser = argparse.ArgumentParser(
        description="Script to wrap VICUNA inference on arbitrary .txt input."
    )

    parser.add_argument(
        "--input_txt",
        required=True,
        help="Input text."
    )

    parser.add_argument(
        "--output_file",
        required=False,
        help="Path to ouput text file.",
        default=None
    )

    args = parser.parse_args()

    run_inference(
        input_file=args.input_file,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()