import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time

st.set_page_config(
    page_title="Question Generator",
    page_icon="❓",
    layout="wide"
)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./model")
    model = AutoModelForSeq2SeqLM.from_pretrained("./model")

    return model, tokenizer

def generate_questions(text, model, tokenizer, num_questions=3):
    input_text = f"generate question: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    output_ids = model.generate(
        input_ids,
        max_length=128,
        min_length=3,
        num_beams=1,
        num_return_sequences=num_questions * 3,
        no_repeat_ngram_size=2,
        temperature=1.1,
        top_k=100,
        top_p=0.98,
        do_sample=True,
        early_stopping=True
    )
    
    questions = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]
    
    processed_questions = []
    seen_questions = set()
    
    for question in questions:
        cleaned_question = question.strip()
        
        if not cleaned_question or len(cleaned_question) < 5:
            continue
            
        if not cleaned_question.endswith("?"):
            cleaned_question += "?"
            
        cleaned_question = cleaned_question.strip('"').strip("'").strip()
        
        lower_q = cleaned_question.lower()
        if lower_q.endswith("what else?") or lower_q.endswith("anything else?"):
            continue
            
        if cleaned_question not in seen_questions:
            seen_questions.add(cleaned_question)
            processed_questions.append(cleaned_question)
            
            if len(processed_questions) >= num_questions:
                break
    
    return processed_questions[:num_questions]


def main():
    st.title("🤖 AI Question Generator")
    st.markdown("""
    This application generates relevant questions based on your input text using AI.
    Simply paste your text below and get insightful questions!
    """)
    
    model, tokenizer = load_model()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter your text here:",
            height=200,
            placeholder="Paste your text here..."
        )
    
    with col2:
        num_questions = st.slider(
            "Number of questions to generate:",
            min_value=1,
            max_value=5,
            value=3
        )
        
    if st.button("Generate Questions", type="primary"):
        if text_input.strip():
            with st.spinner("Generating questions..."):
                try:
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.01)
                    
                    questions = generate_questions(text_input, model, tokenizer, num_questions)
                    
                    st.markdown("### Generated Questions:")
                    
                    questions_container = st.container()
                    
                    for i, question in enumerate(questions, 1):
                        with questions_container:
                            st.markdown(f"**{i}.** {question}")
                    
                    st.success(f"Successfully generated {len(questions)} questions!")
                            
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text first!")

if __name__ == "__main__":
    main()
