from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import re
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1 
)

parser = JsonOutputParser(pydantic_object={
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "card_name": {"type": "string"},
                    "bank_name": {"type": "string"},
                    "payment_system": {"type": "string"},
                    "credit_limit": {"type": "string"},
                    "interest_rate": {"type": "string"},
                    "grace_period": {"type": "string"},
                    "annual_fee": {"type": "string"},
                    "cashback": {"type": "string"},
                    "features": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "benefits": {
                        "type": "object",
                        "properties": {
                            "restaurants": {"type": "string"},
                            "groceries": {"type": "string"},
                            "travel": {"type": "string"},
                            "entertainment": {"type": "string"},
                            "online_shopping": {"type": "string"},
                            "other": {"type": "string"}
                        }
                    }
                },
                "required": ["card_name", "bank_name", "features"]
            }
        }
    },
    "required": ["cards"]
})


prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты эксперт по анализу финансовых продуктов. Твоя задача - извлечь детальную информацию о ВСЕХ КРЕДИТНЫХ КАРТАХ, представленных на странице сравнения, и представить их в структурированном JSON формате.
Проанализируй содержимое страницы и определи все отдельные кредитные карты. Для КАЖДОЙ карты собери следующую информацию:
1. Название карты (card_name)
2. Название банка (bank_name)
3. Платежная система (payment_system): VISA, MasterCard, МИР или UnionPay
4. Кредитный лимит (credit_limit): максимальная сумма кредита
5. Процентная ставка (interest_rate): годовая процентная ставка
6. Льготный период (grace_period): количество дней беспроцентного периода
7. Годовое обслуживание (annual_fee): стоимость обслуживания в год
8. Кэшбэк (cashback): базовый процент кэшбэка
9. Основные особенности карты (features): список из 3-5 ключевых особенностей
10. Бонусы по категориям (benefits): особые условия по разным категориям трат
Формат ответа должен быть следующим:
```json
{{
  "cards": [
    {{
      "card_name": "Название карты 1",
      "bank_name": "Название банка 1",
      "payment_system": "Платежная система",
      "credit_limit": "До X рублей",
      "interest_rate": "X% годовых",
      "grace_period": "До X дней",
      "annual_fee": "X рублей в год",
      "cashback": "X% на все покупки",
      "features": [
        "Особенность 1",
        "Особенность 2",
        "Особенность 3"
      ],
      "benefits": {{
        "restaurants": "X% в ресторанах",
        "groceries": "X% в супермаркетах",
        "travel": "X% на путешествия",
        "entertainment": "X% на развлечения",
        "online_shopping": "X% на онлайн-покупки",
        "other": "Другие специальные условия"
      }}
    }},
    {{
      "card_name": "Название карты 2",
      // ... и так далее для всех карт
    }}
  ]
}}
```json
     
Постарайся идентифицировать и извлечь информацию о как можно большем количестве карт со страницы. Если какая-то информация о карте отсутствует, оставь соответствующее поле пустым или с значением null. Не добавляй информацию, которой нет в исходном тексте."""),
    ("user", "{input}")
])


chain = prompt | llm | parser

def preprocess_html_content(content: str) -> str:
    soup = BeautifulSoup(content, 'html.parser')

    for tag in soup(['script', 'style', 'head', 'footer', 'nav', 'iframe', 'noscript']):
        tag.decompose()
    
    for tag in soup.find_all(True):
        if tag.name != 'img':
            tag.attrs = {}
        else:
            tag.attrs = {k: v for k, v in tag.attrs.items() if k == 'alt'}
    
    text = soup.get_text(separator=' ', strip=True)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    
    print(text)
    
    return text[:6000]

def extract_cards_from_page(url: str = 'https://www.sravni.ru/karty/') -> Dict[str, Any]:
    loader = AsyncChromiumLoader([url])
    html = loader.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html)

    raw_content = docs_transformed[0].page_content

    processed_content = preprocess_html_content(raw_content)
    
    try:
        print("Начинаем анализ данных с помощью модели...")
        result = chain.invoke({"input": processed_content})
        
        if isinstance(result, dict) and "cards" in result:
            cards_count = len(result["cards"])
            print(f"Успешно извлечены данные о {cards_count} картах")
            return result
        else:
            print("Ошибка: неожиданный формат результата")
            return {"cards": []}
            
    except Exception as e:
        print(f"Ошибка при извлечении данных: {e}")
        return {"cards": []}


def save_results(data: Dict[str, Any], filename: str = 'credit_cards.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Результаты сохранены в файл {filename}")

def main():

    result = extract_cards_from_page()

    save_results(result)
    
    if "cards" in result:
        cards = result["cards"]
        print(f"Успешно обработано {len(cards)} карт:")
        for i, card in enumerate(cards):
            print(f"{i+1}. {card.get('card_name', 'Неизвестная карта')} - {card.get('bank_name', 'Неизвестный банк')}")

if __name__ == "__main__":
    main()
