import streamlit as st
import pandas as pd
import json
import requests

# ── Настройки ────────────────────────────────────────────────
API_KEY = "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"
MODEL = "llama-3.1-8b-instant"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ── Инструменты для LLM (tool use) ───────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_basic_stats",
            "description": "Возвращает базовую статистику датасета: количество строк, столбцов, типы данных, пропуски",
            "parameters": {
                "type": "object",
                "properties": {
                    "confirmed": {"type": "boolean", "description": "Подтверди что хочешь получить статистику"}
                },
                "required": ["confirmed"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_numeric_summary",
            "description": "Возвращает среднее, минимум, максимум для числовых столбцов",
            "parameters": {
                "type": "object",
                "properties": {
                    "confirmed": {"type": "boolean", "description": "Подтверди что хочешь получить числовую статистику"}
                },
                "required": ["confirmed"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_categories",
            "description": "Возвращает топ значений для категориальных столбцов",
            "parameters": {
                "type": "object",
                "properties": {
                    "confirmed": {"type": "boolean", "description": "Подтверди что хочешь получить топ категорий"}
                },
                "required": ["confirmed"]
            }
        }
    }
]

# ── Функции-инструменты ───────────────────────────────────────
def get_basic_stats(df):
    result = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0},
        "duplicates": int(df.duplicated().sum())
    }
    return json.dumps(result, ensure_ascii=False)

def get_numeric_summary(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return json.dumps({"message": "Числовые столбцы не найдены"})
    result = {}
    for col in numeric_cols:
        result[col] = {
            "mean": round(float(df[col].mean()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "std": round(float(df[col].std()), 2)
        }
    return json.dumps(result, ensure_ascii=False)

def get_top_categories(df):
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not cat_cols:
        return json.dumps({"message": "Категориальные столбцы не найдены"})
    result = {}
    for col in cat_cols[:5]:
        top = df[col].value_counts().head(3).to_dict()
        result[col] = {str(k): int(v) for k, v in top.items()}
    return json.dumps(result, ensure_ascii=False)

def call_tool(tool_name, df):
    if tool_name == "get_basic_stats":
        return get_basic_stats(df)
    elif tool_name == "get_numeric_summary":
        return get_numeric_summary(df)
    elif tool_name == "get_top_categories":
        return get_top_categories(df)
    return json.dumps({"error": "Инструмент не найден"})

# ── Основная функция анализа через LLM ───────────────────────
def analyze_with_llm(df):
    preview = df.head(5).to_string()
    columns_info = ", ".join([f"{col} ({str(dtype)})" for col, dtype in df.dtypes.items()])

    system_prompt = """Ты — аналитик данных. Тебе дан датасет. 
Используй доступные инструменты чтобы изучить данные, затем напиши аналитический отчёт на русском языке.
Отчёт должен содержать:
1. Краткое описание датасета
2. Ключевые метрики и статистику
3. Интересные наблюдения и тренды
4. Практические выводы и рекомендации"""

    user_message = f"""Проанализируй этот датасет.
Столбцы: {columns_info}
Первые строки:
{preview}

Используй инструменты для получения статистики, затем напиши подробный аналитический отчёт."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Цикл tool use
    for _ in range(5):
        response = requests.post(
            API_URL,
            headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages, "tools": TOOLS, "max_tokens": 2000},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        messages.append(message)

        # Если модель вызывает инструменты
        if choice["finish_reason"] == "tool_calls" and message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_result = call_tool(tool_name, df)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result
                })
        else:
            # Модель закончила — возвращаем текст
            return message.get("content", "Не удалось получить анализ")

    return "Анализ завершён"

# ── Интерфейс Streamlit ───────────────────────────────────────
st.set_page_config(page_title="AI Аналитик данных", page_icon="📊", layout="wide")

st.title("📊 AI Аналитик данных")
st.markdown("Загрузите CSV-файл и получите автоматический анализ от искусственного интеллекта")

uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except Exception:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    st.success(f"Файл загружен: {len(df)} строк, {len(df.columns)} столбцов")

    # Показываем превью
    st.subheader("📋 Превью данных")
    st.dataframe(df.head(10), use_container_width=True)

    # Быстрая статистика
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Строк", len(df))
    with col2:
        st.metric("Столбцов", len(df.columns))
    with col3:
        st.metric("Пропусков", int(df.isnull().sum().sum()))

    st.divider()

    # Кнопка анализа
    if st.button("🤖 Запустить AI-анализ", type="primary", use_container_width=True):
        with st.spinner("AI анализирует данные... Это займёт 10-20 секунд"):
            try:
                result = analyze_with_llm(df)
                st.subheader("📝 Результаты AI-анализа")
                st.markdown(result)

                # Числовая статистика
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) > 0:
                    st.subheader("📈 Числовая статистика")
                    st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

            except Exception as e:
                st.error("Ошибка при анализе: " + str(e))
else:
    st.info("👆 Загрузите CSV-файл чтобы начать анализ")
    st.markdown("""
    **Что умеет это приложение:**
    - 📂 Читает любой CSV-файл
    - 🔍 Анализирует структуру данных с помощью инструментов
    - 📊 Считает ключевые метрики
    - 💡 Формулирует выводы и рекомендации на русском языке
    """)
