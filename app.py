import streamlit as st
import pandas as pd
import re
import io
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smolagents import CodeAgent, LiteLLMModel, tool

# ──────────────────────────────────────────────────────────────────────────
# ЗАЩИТА ОТ PROMPT INJECTION
# ──────────────────────────────────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|previous\s+|above\s+)?instructions",
    r"forget\s+(all\s+|previous\s+|above\s+)?instructions",
    r"you\s+are\s+now", r"\bact\s+as\b",
    r"pretend\s+(you\s+are|to\s+be)", r"\bjailbreak\b",
    r"do\s+anything\s+now", r"(system|admin)\s+(prompt|message|instruction)",
    r"\bdisregard\b", r"\boverride\b",
]

def check_injection(text: str) -> bool:
    if not text:
        return False
    return any(re.search(p, text.lower()) for p in INJECTION_PATTERNS)


# ──────────────────────────────────────────────────────────────────────────
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ──────────────────────────────────────────────────────────────────────────
_current_df: pd.DataFrame = None
_figures_buffer: list = []


# ──────────────────────────────────────────────────────────────────────────
# TOOL: ИНТЕРПРЕТАТОР КОДА
# ──────────────────────────────────────────────────────────────────────────
@tool
def execute_python_code(code: str) -> str:
    """
    Executes Python code for data analysis on the loaded dataset.
    The dataset is available as variable `df` (pandas DataFrame).
    Also available: `pd` (pandas), `plt` (matplotlib.pyplot).
    Use print() for output. Use plt.show() to save charts.
    Returns stdout output as string.

    Args:
        code: Python code to execute
    """
    global _current_df, _figures_buffer

    if _current_df is None:
        return "Error: no dataset loaded"

    forbidden = ["read_csv", "read_excel", "requests.get", "urllib", "open(", "subprocess"]
    for f in forbidden:
        if f in code:
            return f"Error: '{f}' is forbidden. Use existing df variable."

    local_figures = []
    original_show = plt.show

    def capture_show(*args, **kwargs):
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        local_figures.append(buf)
        plt.close(fig)

    plt.show = capture_show

    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code, {
            "df": _current_df.copy(),
            "pd": pd,
            "plt": plt,
        })
        output = sys.stdout.getvalue()
        if not output:
            output = "Code executed successfully (no output)"
    except Exception as e:
        output = f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout
        plt.show = original_show

    _figures_buffer.extend(local_figures)

    if len(output) > 3000:
        output = output[:3000] + "\n...(output truncated)"

    time.sleep(1)

    return output


# ──────────────────────────────────────────────────────────────────────────
# ЗАПУСК АГЕНТА
# ──────────────────────────────────────────────────────────────────────────
def run_agent(df: pd.DataFrame, user_instruction: str = "") -> str:
    global _current_df, _figures_buffer
    _current_df = df
    _figures_buffer = []

    import os
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    model = LiteLLMModel(
        model_id="groq/meta-llama/llama-4-scout-17b-16e-instruct",
        request_timeout=60,
    )

    agent = CodeAgent(
        tools=[execute_python_code],
        model=model,
        max_steps=7,
        additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot", "numpy", "seaborn"],
    )

    columns_info = ", ".join(df.columns.tolist())
    dtypes_info = df.dtypes.to_string()

    task = f"""You are a professional data analyst. You have a tool called `execute_python_code`.

IMPORTANT: You do NOT have direct access to `df`.
You MUST use the `execute_python_code` tool for ALL data access and analysis.
Inside the tool, `df` is a real pandas DataFrame with {len(df)} rows and {len(df.columns)} columns.

Dataset schema:
- Columns: {columns_info}
- Data types:
{dtypes_info}

HOW TO USE THE TOOL:
result = execute_python_code(code=\"\"\"
print(df.describe())
\"\"\")

NEVER define df yourself. NEVER invent data. ALWAYS use execute_python_code tool.
Build charts with plt.show() inside the tool code.
Write the final report in Russian covering: key metrics, distributions, correlations, anomalies, insights.

{"User instruction: " + user_instruction if user_instruction else "Conduct a full exploratory data analysis (EDA)."}
"""

    result = agent.run(task)

    st.session_state.figures = _figures_buffer.copy()

    if not result or str(result).strip() in ["Отчет готов", "None", ""]:
        return "⚠️ Агент завершил анализ. Графики отображены выше."

    return str(result)


# ──────────────────────────────────────────────────────────────────────────
# STREAMLIT ИНТЕРФЕЙС
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Агент-аналитик", layout="wide")
st.title("📊 AI Агент-аналитик данных")

st.markdown("""
**✅ Реализация:**
- Веб-интерфейс на Streamlit
- LLM через Groq API программно (не через браузер)
- Агентный фреймворк: **smolagents** (HuggingFace)
- LLM сама вызывает интерпретатор кода через `@tool`
- Данные не подставляются в промпт — только схема
- Защита от prompt injection
""")

uploaded_file = st.file_uploader(
    "📂 Загрузите CSV или Excel файл",
    type=["csv", "xlsx", "xls"]
)

user_context = st.text_area(
    "📝 Инструкция для агента (необязательно)",
    placeholder="Пример: найди аномалии, построй корреляционную матрицу",
    height=80
)

if user_context and check_injection(user_context):
    st.error("⚠️ Обнаружена попытка изменить поведение системы.")
    st.stop()

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success(f"✅ Загружено: {len(df)} строк, {len(df.columns)} столбцов")

    with st.expander("📋 Превью данных"):
        st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Строк", len(df))
    col2.metric("Столбцов", len(df.columns))
    col3.metric("Пропусков", int(df.isnull().sum().sum()))

    if st.button("Запустить анализ", type="primary", use_container_width=True):
        with st.spinner("Агент анализирует данные..."):
            try:
                report = run_agent(df, user_context)

                st.subheader("📝 Отчёт агента")
                st.markdown(report)

                if st.session_state.get("figures"):
                    st.subheader("📈 Графики")
                    cols = st.columns(min(len(st.session_state.figures), 2))
                    for i, buf in enumerate(st.session_state.figures):
                        buf.seek(0)
                        cols[i % 2].image(buf, use_container_width=True)

            except Exception as e:
                st.error(f"Ошибка: {e}")
else:
    st.info("👆 Загрузите файл для начала анализа")
