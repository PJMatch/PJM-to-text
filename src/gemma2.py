import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "speakleash/Bielik-7B-Instruct-v0.1"

SYSTEM_INSTRUCTION = """Jesteś profesjonalnym tłumaczem Polskiego Języka Migowego (PJM).
Twoim zadaniem jest przekształcenie surowych glosów PJM w poprawne, naturalnie brzmiące zdania w języku polskim.

ZASADY BEZWZGLĘDNE:
1. Zwracaj TYLKO i WYŁĄCZNIE gotowe, przetłumaczone zdanie w języku polskim.
2. NIE dodawaj słowa "Tłumaczenie:".
3. NIE dodawaj żadnych wyjaśnień, komentarzy ani podsumowań.
4. NIE używaj formatowania (np. pogrubień czy gwiazdek).

Przykłady:
Glosy: KOT SPAĆ KANAPA
Kot śpi na kanapie.

Glosy: CHŁOPIEC ROWER JECHAĆ SZYBKO UPŚĆ
Chłopiec jechał szybko na rowerze i upadł.

Zadanie:
Glosy: {glosses}
"""


def main() -> None:
    """
    Main function to load the quantized Bielik model and run 
    real-time PJM translation tests with performance tracking.
    """
    print("Loading and quantizing the model into VRAM (this might take a moment)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )

    print("\n" + "=" * 50)
    print(" SYSTEM READY! ENTER PJM GLOSSES.")
    print(" To exit, type: 'q', 'exit' or 'quit'")
    print("=" * 50 + "\n")

    while True:
        user_input = input("Your PJM glosses: ")
        
        if user_input.lower() in ["q", "exit", "quit"]:
            print("Closing the program...")
            break
            
        if not user_input.strip():
            continue

        prompt_text = SYSTEM_INSTRUCTION.format(glosses=user_input)

        chat = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start_time = time.perf_counter()

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        input_length = inputs.input_ids.shape[-1]
        response = tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )

        end_time = time.perf_counter()
        
        translation_time = end_time - start_time

        print(f"Bielik output:   {response.strip()}")
        print(f"[Response time:  {translation_time:.2f} seconds]")
        print("-" * 50)


if __name__ == "__main__":
    main()