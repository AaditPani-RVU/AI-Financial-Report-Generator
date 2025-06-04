import streamlit as st
import pandas as pd
import io
from transformers import pipeline

# Set up Streamlit
st.set_page_config(page_title="Finance Chatbot", layout="wide")


HF_TOKEN = st.secrets["huggingface"]["token"]
MODEL_NAME = "tiiuae/falcon-7b-instruct"

@st.cache_resource
def load_model():
    return pipeline("text-generation", model=MODEL_NAME, token=HF_TOKEN)


text_gen = load_model()

# ----------- Utility Functions -----------

def chat_with_transactions(prompt, df):
    prompt = prompt.lower()
    months_list = ['january', 'february', 'march', 'april', 'may', 'june',
                   'july', 'august', 'september', 'october', 'november', 'december']

    if ("how much" in prompt or "what is the expenditure" in prompt) and ("spend" in prompt or "expenditure" in prompt):
        for month in months_list:
            if month in prompt:
                return get_monthly_spending(df, month)
        total_expense = df['Amount'].sum()
        return f"You have spent a total of ₹{total_expense:,.2f}."

    if any(x in prompt for x in ["spending on", "spend on", "did i spend on"]):
        for category in df['Category'].dropna().unique():
            if category.lower() in prompt:
                spending = df[df['Category'].str.lower() == category.lower()]['Amount'].sum()
                return f"You have spent ₹{spending:,.2f} on {category.capitalize()}."
        return "I couldn't find that category in your transactions. Try 'groceries', 'entertainment', etc."

    elif any(x in prompt for x in ["top categories", "most spent"]):
        top_categories = df.groupby("Category")["Amount"].sum().sort_values(ascending=False).head(5)
        top_str = "\n".join([f"{i+1}. {cat}: ₹{amt:,.2f}" for i, (cat, amt) in enumerate(top_categories.items())])
        return f"Your top 5 spending categories:\n{top_str}"

    return None

def get_monthly_spending(df, month_name):
    month_df = df[df['Month'].str.lower() == month_name.lower()]
    total = month_df['Amount'].sum()
    return f"You spent ₹{total:,.2f} in {month_name.capitalize()}."

def compare_months(df, month1, month2):
    total1 = df[df['Month'].str.lower() == month1.lower()]['Amount'].sum()
    total2 = df[df['Month'].str.lower() == month2.lower()]['Amount'].sum()
    diff = total1 - total2
    if diff > 0:
        return f"You spent ₹{diff:,.2f} more in {month1.capitalize()} compared to {month2.capitalize()}."
    elif diff < 0:
        return f"You spent ₹{abs(diff):,.2f} less in {month1.capitalize()} compared to {month2.capitalize()}."
    else:
        return f"You spent the same in both months: ₹{total1:,.2f}."



st.title(" Personal Finance Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

uploaded_file = st.file_uploader(" Upload your transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.strftime('%B')

    if 'Category' not in df.columns or 'Amount' not in df.columns:
        st.warning("Please make sure your CSV includes 'Date', 'Amount', and 'Category' columns.")
    else:
        total_expense = df['Amount'].sum()
        top_categories = df['Category'].value_counts().head(5)

        st.subheader(" Quick Summary")
        st.write(f"**Total Transactions:** {len(df)}")
        st.write(f"**Total Amount Spent:** ₹{total_expense:,.2f}")

        st.subheader(" Top Categories")
        st.bar_chart(top_categories)

        report_buffer = io.StringIO()
        report_buffer.write("Personal Finance Report\n")
        report_buffer.write(f"Total transactions: {len(df)}\n")
        report_buffer.write(f"Total spent: ₹{total_expense:,.2f}\n\nTop Categories:\n")
        report_buffer.write(top_categories.to_string())
        report = report_buffer.getvalue()

        st.download_button(" Download Finance Report", data=report, file_name="finance_report.txt")

# ----------- Chat Section -----------

st.subheader(" Chat with your data")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me about your spending, savings, or categories...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    if uploaded_file and 'Amount' in df.columns and 'Category' in df.columns:
        prompt_lower = prompt.lower()
        months_list = ['january', 'february', 'march', 'april', 'may', 'june',
                       'july', 'august', 'september', 'october', 'november', 'december']

        if "compare" in prompt_lower:
            found_months = [m for m in months_list if m in prompt_lower]
            if len(found_months) == 2:
                response = compare_months(df, found_months[0], found_months[1])
            else:
                response = "Please mention two months to compare, like 'Compare March and February'."

        else:
            custom_response = chat_with_transactions(prompt, df)
            if custom_response:
                response = custom_response
            else:
                full_prompt = f"User asked: {prompt}\nRespond like a financial assistant:"
                result = text_gen(full_prompt, max_new_tokens=150, do_sample=True)
                response = result[0]["generated_text"].split("Respond like a financial assistant:")[-1].strip()
    else:
        response = "Please upload a valid CSV first with 'Amount' and 'Category' columns."

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
