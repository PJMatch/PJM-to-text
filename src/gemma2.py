import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b-it"

SYSTEM_INSTRUCTION = """Jesteś profesjonalnym tłumaczem Polskiego Języka Migowego.
Twoim zadaniem jest przekształcenie surowych glosów PJM w poprawne, naturalnie brzmiące i gramatyczne zdania w języku polskim.

Przykłady:
Glosy: KOT SPAĆ KANAPA
Tłumaczenie: Kot śpi na kanapie.

Glosy: CHŁOPIEC ROWER JECHAĆ SZYBKO UPŚĆ
Tłumaczenie: Chłopiec jechał szybko na rowerze i upadł.

Zadanie:
Glosy: {glosses}
Tłumaczenie:"""

TEST_CASES = [
    "JA JUTRO KINO IŚĆ CHCEĆ",
    "KOBIETA PIES WIDZIEĆ UCIEKAĆ BAĆ SIĘ",
    "TY KSIĄŻKA CZYTAĆ LUBIĆ CZY",
    "MÓJ BRAT SAMOCHÓD NOWY KUPIĆ WCZORAJ",
    "DZIECKO PŁAKAĆ BO ZABAWKA ZEPSUĆ",
]


def main() -> None:
    """
    Main function to initialize the Gemma model and run PJM translation tests.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("\n--- STARTING TESTS ---\n")

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
            max_new_tokens=50,
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