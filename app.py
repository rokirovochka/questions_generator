import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import time

st.set_page_config(
    page_title="Question Generator",
    page_icon="❓",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = "mirfan899/t5-e2e-questions-generation"
    st.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

def generate_questions(text, model, tokenizer, num_questions=3):
    input_text = f"generate questions: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    output_ids = model.generate(
        input_ids,
        max_length=150,
        min_length=3,  
        num_beams=5,
        num_return_sequences=num_questions,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True, 
        early_stopping=True
    )
    
    questions = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]
    
    processed_questions = []
    for question in questions:
        if question.endswith("what else?") or question.endswith("what else"):
            continue
        
        if question.count("?") > 1:
            split_questions = [q.strip() + "?" for q in question.split("?") if q.strip()]
            processed_questions.extend(split_questions)
        else:
            if not question.strip().endswith("?"):
                question = question.strip() + "?"
                
            if question.strip():
                processed_questions.append(question)
    
    if not processed_questions:
        return questions
    
    if len(processed_questions) > num_questions:
        processed_questions = processed_questions[:num_questions]
    elif len(processed_questions) < num_questions:
        while len(processed_questions) < num_questions:
            processed_questions.append(processed_questions[0])
    
    return processed_questions

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