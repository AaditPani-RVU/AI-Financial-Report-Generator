import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import pandas as pd
import io

# ----------- Chat Functions -----------

# Load model and tokenizer manually (Changed to GPT2 or GPT-NEO for CausalLM)
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Changed to GPT-2 for causal language modeling
model = AutoModelForCausalLM.from_pretrained("gpt2")  # Changed to GPT-2 for causal language modeling

def chat_with_transactions(prompt, df):
    prompt = prompt.lower()

    months_list = ['january', 'february', 'march', 'april', 'may', 'june',
                   'july', 'august', 'september', 'october', 'november', 'december']

    # Total spend
    if ("how much" in prompt or "what is the expenditure" in prompt) and "spend" in prompt or "expenditure" in prompt:
        # Month-based check
        for month in months_list:
            if month in prompt:
                return get_monthly_spending(df, month)
        total_expense = df['Amount'].sum()
        return f"You have spent a total of ₹{total_expense:,.2f}"

    # Category-based spend
    if "spending on" in prompt or "spend on" in prompt or "did i spend on" in prompt:
        for category in df['Category'].unique():
            if category.lower() in prompt:
                spending = df[df['Category'].str.lower() == category.lower()]['Amount'].sum()
                return f"You have spent ₹{spending:,.2f} on {category.capitalize()}."
        return "I couldn't find that category in your transactions. Try something else like 'groceries', 'entertainment', etc."

    # Top categories
    elif "top categories" in prompt or "most spent" in prompt:
        top_categories = df['Category'].value_counts().head(5)
        return f"Your top 5 spending categories are:\n{top_categories}"

    return "I'm not sure how to answer that. Try asking about 'total spend', 'top categories', or 'spending on groceries.'"


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


# ----------- Streamlit Setup -----------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.set_page_config(page_title="Finance Chatbot", layout="wide")
st.title("💬 Personal Finance Chatbot")

# ----------- Upload CSV -----------

uploaded_file = st.file_uploader("📂 Upload your transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Parse Date column and add Month
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.strftime('%B')

    total_expense = df['Amount'].sum()
    top_categories = df['Category'].value_counts().head(5)

    st.subheader("📊 Quick Summary")
    st.write(f"**Total Transactions:** {len(df)}")
    st.write(f"**Total Amount Spent:** ₹{total_expense:,.2f}")

    st.subheader("📌 Top Categories")
    st.bar_chart(top_categories)

    # Download Report
    report_buffer = io.StringIO()
    report_buffer.write("Personal Finance Report\n")
    report_buffer.write(f"Total transactions: {len(df)}\n")
    report_buffer.write(f"Total spent: ₹{total_expense:,.2f}\n")
    report_buffer.write("\nTop Categories:\n")
    report_buffer.write(top_categories.to_string())
    report = report_buffer.getvalue()

    st.download_button("📄 Download Finance Report", data=report, file_name="finance_report.txt")

# ----------- Chat Interface -----------

st.subheader("💬 Chat with your data")

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me about your spending, savings, or categories...")

# ----------- Chat Logic -----------

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    if uploaded_file:
        months_list = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                       'august', 'september', 'october', 'november', 'december']
        prompt_lower = prompt.lower()

        if "compare" in prompt_lower:
            found_months = [m for m in months_list if m in prompt_lower]
            if len(found_months) == 2:
                response = compare_months(df, found_months[0], found_months[1])
            else:
                response = "Please mention two months to compare, like 'Compare March and February'." 

        elif "spend" in prompt_lower and any(m in prompt_lower for m in months_list):
            found_month = next((m for m in months_list if m in prompt_lower), None)
            if found_month:
                response = get_monthly_spending(df, found_month)
            else:
                response = "Please mention a valid month to get your spending."

        elif "groceries" in prompt_lower:
            response = chat_with_transactions(prompt, df)

        else:
            # Tokenize and generate response (Causal LM task)
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(inputs["input_ids"], max_length=300, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    else:
        response = "Please upload a CSV first so I can analyze your transactions."

    # To prevent repetition, check if the response is identical to the last response
    if not st.session_state["messages"] or st.session_state["messages"][-1]["content"] != response:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
