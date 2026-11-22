"""DSC670 Project Milestone 4 - Telecom Plan and Billing Assistant.

This Streamlit application loads the full fine-tuned Qwen2.5-0.5B-Instruct
checkpoint created in Milestone 3 and generates evidence-grounded telecom plan
selection and billing assistance.

Expected model folder (relative to this file):
    milestone3_outputs/qwen_full_finetune/

Run:
    pip install -r requirements.txt
    streamlit run Download\ the\ Streamlit\ application.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "milestone3_outputs" / "qwen_full_finetune"
SYSTEM_PROMPT = (
    "You are a telecom plan and billing assistant. Use only facts supplied by the user. "
    "Do not invent prices, fees, promotions, taxes, policies, account activity, availability, "
    "or guarantees. When required facts are missing, ask concise follow-up questions. "
    "Structure the answer with: Customer need, Recommendation or explanation, Evidence, "
    "Tradeoff or uncertainty, and Next step."
)

st.set_page_config(page_title="Telecom Plan & Billing Assistant", page_icon="📡", layout="wide")

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg,#071a2d 0%,#0e2941 55%,#123a4a 100%);}
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    h1,h2,h3,p,label,.stMarkdown {color:#eef8fb !important;}
    .hero {padding:1.3rem 1.5rem;border-radius:18px;background:linear-gradient(120deg,#0e6070,#14345e);border:1px solid #3bd5ce;box-shadow:0 12px 30px rgba(0,0,0,.25);}
    .metric-card {padding:1rem;border-radius:14px;background:#102f48;border:1px solid #2c7080;min-height:108px;}
    .safe {padding:.8rem 1rem;border-left:5px solid #33d1c6;background:#12334b;border-radius:8px;}
    .stButton>button {background:#f28c79;color:#071a2d;border:0;border-radius:10px;font-weight:700;}
    .stTextArea textarea {background:#f5fbfd;color:#071a2d;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📡 Telecom Plan & Billing Assistant</h1>
        <p>Fine-tuned Qwen2.5 model for grounded plan selection, billing explanations, and safe follow-up questions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
### Why This Assistant is Reliable

✓ **Grounded in Facts** - Uses only information you provide, no invented prices or guarantees  
✓ **Safety First** - Asks for missing information rather than guessing  
✓ **Clear Structure** - Provides transparent reasoning in every response  
✓ **Tabular Data** - Displays plans and comparisons in easy-to-read tables when needed  

---
""")

with st.sidebar:
    st.header("Model configuration")
    model_path_text = st.text_input("Fine-tuned model folder", str(DEFAULT_MODEL_PATH))
    max_new_tokens = st.slider("Maximum response tokens", 100, 500, 300, 25)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    st.caption("For repeatable academic demonstrations, keep temperature near 0.2.")
    st.divider()
    st.subheader("Milestone 3 evidence")
    st.write("Model: Qwen/Qwen2.5-0.5B-Instruct")
    st.write("Method: Full-parameter supervised fine-tuning")
    st.write("Parameters updated: 494,032,768 (100%)")
    st.write("Training/validation: 42 / 11 records")
    st.write("Held-out score: 2.36 → 3.64")

MODEL_PATH = Path(model_path_text).expanduser()
DEMO_MODE = os.getenv("TELECOM_APP_DEMO", "0") == "1"

@st.cache_resource(show_spinner="Loading fine-tuned Qwen model...")
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, local_files_only=True)
    model.to(device)
    model.eval()
    return model, tokenizer, device

def looks_telecom_related(text: str) -> bool:
    terms = r"plan|internet|broadband|wifi|speed|device|data|bill|billing|charge|credit|fee|promotion|prorat|account|service|router|modem|mobile|roam|network"
    return bool(re.search(terms, text, flags=re.I))

def demo_response(prompt: str) -> str:
    if "bill" in prompt.lower() and "$20" in prompt:
        return """Customer need\nThe customer wants to understand a $20 bill increase.\n\nRecommendation or explanation\nThe supplied $20 promotional-credit change explains the increase.\n\nEvidence\nThe bill moved from $80 to $100, and the prompt states that a $20 promotional credit ended.\n\nTradeoff or uncertainty\nNo other charge changes were supplied, so no additional cause can be confirmed.\n\nNext step\nReview the promotional-credit line item and contact billing support if the change was unexpected."""
    return """Customer need\nGuidance is requested, but verified facts are missing.\n\nRecommendation or explanation\nA specific plan cannot be confirmed yet.\n\nEvidence\nThe request does not include available plans, prices, device count, usage, or budget.\n\nTradeoff or uncertainty\nGuessing could create an unsupported recommendation.\n\nNext step\nProvide the available plan options, monthly budget, device count, and major usage needs."""

def generate_answer(user_text: str) -> tuple[str, str]:
    if DEMO_MODE:
        return demo_response(user_text), "Interface preview mode"
    model, tokenizer, device = load_model(str(MODEL_PATH))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    kwargs = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.08, pad_token_id=tokenizer.eos_token_id)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return answer, f"Fine-tuned checkpoint on {device.upper()}"

st.markdown("---")
st.subheader("Quick Questions")

col1, col2, col3 = st.columns(3)

question_1 = "My bill increased from $80 to $100. The bill states that a $20 promotional credit ended, and no other charge changed. Explain the increase."
question_2 = "Available plans: Connect 200 costs $45 with 200 Mbps download and 10 Mbps upload; Work 500 costs $70 with 500 Mbps download and 50 Mbps upload; Gig 1000 costs $95 with 1000 Mbps download and 100 Mbps upload. I work from home, use 10 devices, attend 8 hours of video calls daily, and my budget is $75. Recommend the strongest eligible option."
question_3 = "A family of four streams 4K video on three televisions, has two online gamers, uses approximately 900 GB of data per month, and wants to stay below $110. Compare: Family 500 at $85 with 1.2 TB data; Family Gig at $105 with unlimited data; and Premium Gig at $130 with unlimited data and equipment bundle. Recommend one option and explain the tradeoff."

with col1:
    if st.button("📊 Bill Increase?", use_container_width=True):
        st.session_state.selected_question = question_1
        st.session_state.show_answer = True
        st.rerun()

with col2:
    if st.button("💼 Which Plan?", use_container_width=True):
        st.session_state.selected_question = question_2
        st.session_state.show_answer = True
        st.rerun()

with col3:
    if st.button("👨‍👩‍👧‍👦 Family Plans?", use_container_width=True):
        st.session_state.selected_question = question_3
        st.session_state.show_answer = True
        st.rerun()

if st.session_state.get("show_answer") and st.session_state.get("selected_question"):
    if looks_telecom_related(st.session_state.selected_question):
        try:
            with st.spinner("Generating response from the fine-tuned model..."):
                answer, source = generate_answer(st.session_state.selected_question)
            st.success(source)
            st.markdown("### 💡 Answer")
            st.markdown(answer)
            st.markdown('<div class="safe">Academic proof of concept. Verify current plans, policies, and account changes with approved provider systems before acting.</div>', unsafe_allow_html=True)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("Copy the saved Milestone 3 checkpoint to milestone3_outputs/qwen_full_finetune, or update the model folder in the sidebar.")
        except Exception as exc:
            st.error(f"The model could not generate a response: {exc}")
    st.session_state.show_answer = False
    st.markdown("---")

st.markdown("### Or Ask Your Own Question")
user_question = st.text_area(
    "Enter your question", 
    value=st.session_state.get("selected_question", ""), 
    height=120,
    placeholder="Include verified plan options, prices, budget, usage, or bill facts...",
    key="user_input"
)

if st.button("Get Answer 🚀", use_container_width=True):
    if not user_question.strip():
        st.warning("Please enter your question.")
    elif not looks_telecom_related(user_question):
        st.warning("This application is limited to telecom plan, service, and billing questions.")
    else:
        try:
            with st.spinner("Generating response from the fine-tuned model..."):
                answer, source = generate_answer(user_question)
            st.success(source)
            st.markdown("### 💡 Answer")
            st.markdown(answer)
            st.markdown('<div class="safe">Academic proof of concept. Verify current plans, policies, and account changes with approved provider systems before acting.</div>', unsafe_allow_html=True)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("Copy the saved Milestone 3 checkpoint to milestone3_outputs/qwen_full_finetune, or update the model folder in the sidebar.")
        except Exception as exc:
            st.error(f"The model could not generate a response: {exc}")

with st.expander("How the application protects groundedness"):
    st.markdown("""
    1. The fine-tuned checkpoint receives the same evidence-boundary system prompt used during training.
    2. The interface asks users to include verified plan or billing facts.
    3. Non-telecom questions are rejected before model inference.
    4. The response uses five sections so evidence and uncertainty can be reviewed.
    5. The application displays a human-verification reminder because the model is not production-ready.
    """)

st.caption("Akhil Anand Thazhaniyil Muralidharan | DSC670-T301 | Bellevue University | Project Milestone 4")
