import re
import os
import pandas as pd
import numpy as np
import json
from langchain.schema import Document
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from uuid import uuid4
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import fcntl
import time
import ast

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class VectorDatabaseManager:
    def __init__(self, model_name):
        self.model = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None
        self.client = None
        
        self.stop_words = set(stopwords.words('russian') + stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words=list(self.stop_words))
        
        # Add a lock file path
        self.lock_file_path = "data/vector_database_lock"
        
    def __del__(self):
        """Clean up resources when object is destroyed"""
        self.close()
    
    def close(self):
        """Explicitly close the client connection"""
        if self.client is not None:
            try:
                self.client.close()
                self.client = None
                self.vectorstore = None
                if os.path.exists(self.lock_file_path):
                    os.remove(self.lock_file_path)
            except Exception as e:
                print(f"Error closing client: {e}")
        
    def load_database(self):
        """Load existing vector database with lock handling"""
        if os.path.exists(self.lock_file_path):
            import time
            for attempt in range(3):
                time.sleep(1)
                if not os.path.exists(self.lock_file_path):
                    break
                if attempt == 2:
                    try:
                        os.remove(self.lock_file_path)
                    except:
                        pass
        
        try:
            with open(self.lock_file_path, 'w') as f:
                f.write(str(os.getpid()))
        except Exception as e:
            print(f"Could not create lock file: {e}")
            
        try:
            self.client = QdrantClient(path="data/vector_database")
            self.vectorstore = Qdrant(
                client=self.client,
                collection_name="data",
                embeddings=self.model
            )
            
            if os.path.exists("data/vector_database/vectorizer_vocabulary.npy"):
                self.vocabulary = np.load("data/vector_database/vectorizer_vocabulary.npy", allow_pickle=True)
                self.vectorizer.vocabulary_ = {word: i for i, word in enumerate(self.vocabulary)}
        except Exception as e:
            print(f"Could not load database: {e}")
            self.client = None
            self.vectorstore = None
        
    def search_VB(self, query, k=5, filters=None):
        """
        Функция поиска по векторной базе
        query - запрос
        k - кол-во документов в выдачи
        filters - словарь фильтров для поиска карт
        На выходе строка содержащая информацию из документов для модели
        """
        results = []
        if self.vectorstore is None:
            self.load_database()
        
        for doc, score in self.vectorstore.similarity_search_with_score(query, k=k):
            print(score, doc, '\n' * 3)
            
            metadata = doc.metadata
            
            # Преобразуем содержимое карты из строки в словарь если это возможно
            card_data = doc.page_content
            try:
                card_dict = ast.literal_eval(card_data)
                
                # Применяем фильтры если они есть
                if filters and isinstance(card_dict, dict):
                    skip_card = False
                    
                    # Проверяем каждый фильтр
                    for filter_key, filter_value in filters.items():
                        # Специальная обработка для вложенных полей
                        if filter_key == 'benefits' and 'benefits' in card_dict:
                            # Проверяем наличие указанного преимущества
                            benefit_found = False
                            for benefit_category, benefit_value in card_dict['benefits'].items():
                                if benefit_value and filter_value.lower() in str(benefit_value).lower():
                                    benefit_found = True
                                    break
                            if not benefit_found:
                                skip_card = True
                                break
                        # Проверка для обычных полей
                        elif filter_key in card_dict:
                            card_value = card_dict[filter_key]
                            if card_value is None or filter_value.lower() not in str(card_value).lower():
                                skip_card = True
                                break
                    
                    if skip_card:
                        continue
            except:
                # Если не удалось преобразовать в словарь, используем как есть
                pass
            
            result = {
                "file": metadata.get('source_file', ''),
                "content": doc.page_content,
                'score': score
            }
            results.append(result)
            if len(results) >= k:
                break
                
        if hasattr(self, 'vocabulary') and self.vectorizer.vocabulary_:
            try:
                query_terms = query.lower().split()
                documents_with_scores = []
                
                all_docs = self.client.scroll(
                    collection_name="data",
                    limit=1000,  
                    with_payload=True
                )[0]
                
                for doc in all_docs:
                    text = doc.payload.get("page_content", "")
                    
                    # Применяем фильтры, если они есть
                    if filters:
                        try:
                            card_dict = ast.literal_eval(text)
                            skip_card = False
                            
                            for filter_key, filter_value in filters.items():
                                # Специальная обработка для вложенных полей
                                if filter_key == 'benefits' and 'benefits' in card_dict:
                                    benefit_found = False
                                    for benefit_category, benefit_value in card_dict['benefits'].items():
                                        if benefit_value and filter_value.lower() in str(benefit_value).lower():
                                            benefit_found = True
                                            break
                                    if not benefit_found:
                                        skip_card = True
                                        break
                                # Проверка для обычных полей
                                elif filter_key in card_dict:
                                    card_value = card_dict[filter_key]
                                    if card_value is None or filter_value.lower() not in str(card_value).lower():
                                        skip_card = True
                                        break
                            
                            if skip_card:
                                continue
                        except:
                            pass
                    
                    score = 0
                    for term in query_terms:
                        if term in text.lower():
                            term_freq = text.lower().count(term) / len(text.split())
                            if term in self.vectorizer.vocabulary_:
                                score += term_freq * (1.0 / (self.vectorizer.vocabulary_.get(term, 1) + 1))
                    
                    if score > 0:
                        documents_with_scores.append({
                            "file": doc.payload.get("source_file", ""),
                            "content": text,
                            "score": score
                        })
                
                keyword_results = sorted(documents_with_scores, key=lambda x: x["score"], reverse=True)[:k]
                
                existing_contents = {r["content"] for r in results}
                for kr in keyword_results:
                    if kr["content"] not in existing_contents:
                        results.append(kr)
                        existing_contents.add(kr["content"])
                
                results = results[:k]
            except Exception as e:
                print(f"Error in keyword search: {e}")
                
        return results
    
    def VB_build(self, csv_path):
        """
        Функция создает с нуля векторную базу из CSV файла с данными о картах
        csv_path - путь к CSV файлу с данными
        """
        df = pd.read_csv(csv_path)
        
        # Подготавливаем тексты для векторизатора
        texts = []
        for _, row in df.iterrows():
            if 'cards' in df.columns:
                # Предполагаем, что в столбце 'cards' содержатся словари в виде строк
                try:
                    card_data = ast.literal_eval(row['cards'])
                    # Преобразуем все значения словаря в строки для обработки
                    text = " ".join([str(v) for v in card_data.values() if v])
                    texts.append(text)
                except:
                    texts.append(str(row['cards']))
        
        # Обучаем векторизатор на текстах
        self.vectorizer.fit(texts)
        self.vocabulary = np.array(self.vectorizer.get_feature_names_out())
        
        os.makedirs("data/vector_database", exist_ok=True)
        np.save("data/vector_database/vectorizer_vocabulary.npy", self.vocabulary)
        
        self.client = QdrantClient(path="data/vector_database")
        
        if self.client.collection_exists("data"):
            self.client.delete_collection("data")
            
        self.client.create_collection(
            collection_name='data', 
            vectors_config=VectorParams(size=len(self.model.encode("test")), distance=Distance.COSINE)
        )
        
        documents = []
        for _, row in df.iterrows():
            if 'cards' in df.columns:
                # Используем данные карт как содержимое документа
                documents.append(
                    Document(
                        page_content=row['cards'],
                        metadata={'source_file': f"card_{_}"}
                    )
                )
        
        self.vectorstore = Qdrant(
            client=self.client,
            collection_name='data',
            embeddings=self.model
        )
        
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.vectorstore.add_documents(documents=batch)
        
        print('Vector store created with BM25-like search capability')
        
        return True
    
    def insert(self, new_documents):
        """Add new documents to the database"""
        if self.vectorstore is None:
            self.load_database()
            
        self.vectorstore.add_documents(documents=new_documents)
        
        if hasattr(self, 'vocabulary'):
            try:
                new_texts = [doc.page_content for doc in new_documents]
                self.vectorizer.fit(new_texts)
                new_vocabulary = set(self.vectorizer.get_feature_names_out())
                
                current_vocab = set(self.vocabulary)
                updated_vocab = current_vocab.union(new_vocabulary)
                self.vocabulary = np.array(list(updated_vocab))
                
                np.save("data/vector_database/vectorizer_vocabulary.npy", self.vocabulary)
            except Exception as e:
                print(f"Error updating vectorizer: {e}")
            
        return True
    
    def preprocess_text(self, text):
        '''
        Функция препроцессит текст чанка
        text - путь выбранной директории
        Выход: датафрейм чанков
        '''
        text = re.sub(r'[^a-zA-Zа-яА-Я0-9]', ' ', text)
        text = text.lower()
        text = text.strip()
        return text
    
    def dataframe_preprocess(self, df):
        '''
        Функция препроцессинга датафрейма чанков
        df - датафрейм
        Выход: обновленный датафрейм
        '''
        try:
            df = df.drop(columns=['Unnamed: 0'])
        except Exception:
            pass
            
        df = df.dropna().reset_index(drop=True)
        
        # Если есть столбец cards, используем его как chunk_text
        if 'cards' in df.columns:
            df['chunk_text'] = df['cards']
        
        df['preprocessed_text'] = df['chunk_text'].apply(lambda x: self.preprocess_text(x))
        df['embedded_text'] = df['preprocessed_text'].apply(lambda x: self.model.embed_query(x))
        print('Preprocessing completed')
        return df
    
    def dir_parse(self, output_dir: str):
        '''
        Функция парсит выбранную директорию в датафрейм чанков
        output_dir - путь выбранной директории
        Выход: датафрейм чанков
        '''
        def chunk_splitter(markdown_text):
            chunked_text = markdown_text.split('#')[1::]
            return chunked_text
            
        md_files = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
        chunks = dict()
        
        for i in md_files:
            try:
                with open(output_dir + '/' + i + '/' + re.sub(r'\.docx$|\.doc$|\.pdf$', '', i, count=1) + '.md', 'r') as f:
                    md_text = f.read()
                    chunks[i] = chunk_splitter(md_text)
            except Exception:
                print(output_dir + '/' + i + '/' + i + '.md')
                
        for i in list(chunks.keys()):
            if chunks[i] == []:
                chunks.pop(i)
        
        rows = []
        for source_file, chunk in chunks.items():
            for chunk_text in chunk:
                if len(chunk_text) < 100:
                    continue
                rows.append({'source_file': source_file, 'chunk_text': chunk_text})
        return pd.DataFrame(rows)

if __name__ == '__main__':
    vb = VectorDatabaseManager('ai-forever/sbert_large_nlu_ru')
    
    # Пример использования с фильтрами
    vb.VB_build('data/sravni_cards.csv')
    
    # Поиск без фильтров
    results = vb.search_VB('кредитная карта с кэшбэком')
    print("Результаты без фильтров:")
    for r in results:
        print(f"Score: {r['score']}")
        print(f"Content: {r['content'][:100]}...")
        print("-" * 50)
    
    # Поиск с фильтрами
    filters = {
        'bank_name': 'Сбербанк',
        'benefits': 'путешествия'
    }
    filtered_results = vb.search_VB('кредитная карта с кэшбэком', filters=filters)
    print("\nРезультаты с фильтрами:")
    for r in filtered_results:
        print(f"Score: {r['score']}")
        print(f"Content: {r['content'][:100]}...")
        print("-" * 50)