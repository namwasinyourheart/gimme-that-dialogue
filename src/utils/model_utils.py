import torch
from transformers import BitsAndBytesConfig, PreTrainedTokenizer, AutoTokenizer


def load_tokenizer(data_args, model_args,
                  # padding_side
) -> PreTrainedTokenizer:
    tokenizer =  AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
    if not tokenizer.pad_token:
        
        if data_args.tokenizer.new_pad_token:
            tokenizer.padding_side = 'left'
            tokenizer.pad_token = data_args.tokenizer.new_pad_token,
            tokenizer.add_special_tokens({"pad_token": data_args.tokenizer.new_pad_token})
        else:
            tokenizer.padding_side = 'left'
            tokenizer.pad_token = tokenizer.eos_token
    
    if data_args.tokenizer.add_special_tokens:
        additional_special_tokens = [
            data_args.prompt.instruction_key, 
            data_args.prompt.input_key,  
            data_args.prompt.response_key
        ]
        
        if data_args.prompt.context_key:
            additional_special_tokens = additional_special_tokens.append(data_args.prompt.context_key)
            
        tokenizer.add_special_tokens({
            "additional_special_tokens": additional_special_tokens
        })

    tokenizer.padding_side = 'left'
                
    return tokenizer

def set_torch_dtype_and_attn_implementation():
    # Set torch dtype and attention implementation
    try:
        if torch.cuda.get_device_capability()[0] >= 8:
            # !pip install -qqq flash-attn
            torch_dtype = torch.bfloat16
            attn_implementation = "flash_attention_2"
        else:
            torch_dtype = torch.float16
            attn_implementation = "eager"
    except:
        torch_dtype = torch.float16
        attn_implementation = "eager"

    return torch_dtype, attn_implementation



def get_quantization_config(model_args                          
) -> BitsAndBytesConfig | None:
    if model_args['load_in_4bit']:
        torch_dtype, attn_implementation = set_torch_dtype_and_attn_implementation()
        
        if model_args['bnb_4bit_compute_dtype']:
            bnb_4bit_compute_dtype = model_args['bnb_4bit_compute_dtype']
        else:
            bnb_4bit_compute_dtype = torch_dtype

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=model_args['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=model_args['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_storage=model_args['bnb_4bit_quant_storage'],
        ).to_dict()
    elif model_args['load_in_8bit']:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        ).to_dict()
    else:
        quantization_config = None

    return quantization_config