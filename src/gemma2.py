import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "google/gemma-2-2b-it"

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

TEST_CASES = [
    "JA JUTRO KINO IŚĆ CHCEĆ",
    "KOBIETA PIES WIDZIEĆ UCIEKAĆ BAĆ-SIĘ",
    "TY KSIĄŻKA CZYTAĆ LUBIĆ CZY",
    "MÓJ BRAT SAMOCHÓD NOWY KUPIĆ WCZORAJ",
    "DZIECKO PŁAKAĆ BO ZABAWKA ZEPSUĆ",
]


def main() -> None:
    """
    Main function to load the quantized Gemma model and run PJM translation tests locally.
    """
    
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

    print("\dziala\n")

    for gloss in TEST_CASES:
        prompt_text = SYSTEM_INSTRUCTION.format(glosses=gloss)

        chat = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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

        print(f"Input glosses: {gloss}")
        print(f"Gemma output:  {response.strip()}")
        print("-" * 40)


if __name__ == "__main__":
    main()