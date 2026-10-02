# practice-4
## Text summarization

Проект посвящен сравнению эффективности LoRA и Zero-shot подходов на моделях **Qwen 3.5 0.8B** и **Qwen 3.5 27B** для задачи суммаризации диалогов.

---


## 1. EDA

- **Качество данных:** Пропуски и дубликаты в исходных выборках отсутствуют.
- **Распределение длин диалогов:** Большинство диалогов содержит **от 85 до 163 слов**, однако присутствуют длинные выбросы **до 985 слов**.
- **Распределение длин целевых саммари:** Саммари значительно короче исходных диалогов — в среднем **около 23 слов**.
- **Особенности задачи:** Задача предполагает существенное сжатие исходного текста. Распределение длин имеет выраженный длинный хвост, что необходимо учитывать при выборе длины контекста (`max_length = 512`).

---

## 2. Проделанная работа

1. **Baseline:** Выделение первого и последнего предложений диалога.
2. **SFT Fine-Tuning (LoRA):** Проведено LoRA-обучение моделей **Qwen 3.5 0.8B** и **Qwen 3.5 27B** 
3. **Инференс:** Настроен быстрый инференс через vLLM  для Zero-shot и LoRA версий.
5. **Валидация и оценка:**
   - Расчёт классических метрик **BLEU** и **ROUGE-1/2/L**.
   - Настройка оценки **LLM-as-a-Judge** через Groq API по 3 критериям (*Faithfulness*, *Relevance*, *Conciseness*).

---

## 3. Результаты

### Автоматические метрики (BLEU & ROUGE)

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** | 2.7270 | 0.1640 | 0.0451 | 0.1401 |
| **Qwen 0.8B Zero-shot** | 4.4864 | 0.3411 | 0.0916 | 0.2522 |
| **Qwen 0.8B LoRA** | 14.4973 | 0.4143 | 0.1604 | 0.3252 |
| **Qwen 27B Zero-shot** | 5.6649 | 0.3546 | 0.1137 | 0.2751 |
| **Qwen 27B LoRA** | **19.2549** | **0.4236** | **0.1834** | **0.3551** |

### LLM-as-a-Judge Summary (Шкала 1–5)

| Model | Faithfulness | Relevance | Conciseness | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen 27B Zero-shot** | **4.104** | **4.064** | **4.052** | **12.220** |
| **Qwen 27B LoRA** | 4.059 | 4.017 | 4.017 | 12.094 |
| **Qwen 0.8B LoRA** | 4.052 | 4.020 | 4.007 | 12.079 |
| **Qwen 0.8B Zero-shot** | 4.057 | 3.998 | 3.998 | 12.052 |
| **Baseline** | 3.535 | 3.507 | 3.505 | 10.547 |

---

## 4. Выводы

1. **Эффект Fine-Tuning (LoRA):**
   - Тонкая настройка дает мощнейший прирост по классическим n-gram метрикам. BLEU вырастает с **4.48 до 14.50** на модели 0.8B и с **5.66 до 19.25** на модели 27B.
   - LoRA эффективно обучает модель повторять стиль и формат целевых референсов из датасета.

2. **Влияние размера модели:**
   - Модель **Qwen 3.5 27B LoRA** показала лидерство по BLEU (19.25) и ROUGE-L (0.3551).

3. **LLM-as-a-Judge vs N-gram метрики:**
   - По оценкам судейской модели **Qwen 27B Zero-shot** набрала наивысший общий балл (12.220). Большие модели «из коробки» генерируют более естественные ответы для человека, даже если их формулировки отличаются от эталонных CSV-файлов (что занижает их BLEU).
   - Обученная **Qwen 0.8B LoRA** по качеству судейской оценки (12.079) практически сравнялась с **Qwen 27B LoRA** (12.094), что делает 0.8B LoRA отличным выбором для проектов с ограниченными GPU-ресурсами.

---

## 5. Инструкция по запуску и CLI-интерфейс

### 1. Подготовка окружения

pip install -r requirements.txt

### 2. Для работы этапа оценки через LLM-as-a-Judge

export GROQ_API_KEY="your_groq_api_key_here"

### 3. Проверка данных и генерация Baseline предсказаний

python train_pipeline.py --stage prepare
python train_pipeline.py --stage baseline

### 4. Запуск экспериментов для Qwen 3.5 0.8B
python train_pipeline.py --model-config configs/model_qwen08b.yaml --stage zero_shot
  
python train_pipeline.py --model-config configs/model_qwen08b.yaml --stage train
  
python train_pipeline.py --model-config configs/model_qwen08b.yaml --stage lora

### 5. Запуск экспериментов для Qwen 3.5 27B
python train_pipeline.py --model-config configs/model_qwen27b.yaml --stage zero_shot
  
python train_pipeline.py --model-config configs/model_qwen27b.yaml --stage train
  
python train_pipeline.py --model-config configs/model_qwen27b.yaml --stage lora

### 6. Итоговая оценка всех 5 моделей (BLEU, ROUGE и LLM-as-a-Judge)
python train_pipeline.py --stage evaluate